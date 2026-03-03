import wandb
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR

from qm9.dataset import QM9
from qm9.evaluate import evaluate
import utils
import time


def train(gpu, model, args, global_epoch):
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

    if args.gpus == 0:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
        if args.gpus > 1:
            dist.init_process_group("nccl", rank=gpu, world_size=args.gpus)
            torch.cuda.set_device(gpu)

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    torch._dynamo.config.capture_scalar_outputs = True
    model = torch.compile(model)

    train_loader = utils.make_dataloader(
        QM9(args.root, args.target, args.radius, "train", args.lmax_attr, feature_type=args.feature_type),
        args.batch_size,
        args.num_workers,
        args.gpus,
        gpu,
    )
    valid_loader = utils.make_dataloader(
        QM9(args.root, args.target, args.radius, "valid", args.lmax_attr, feature_type=args.feature_type),
        args.batch_size,
        args.num_workers,
        args.gpus,
        gpu,
        train=False,
    )
    test_loader = utils.make_dataloader(
        QM9(args.root, args.target, args.radius, "test", args.lmax_attr, feature_type=args.feature_type),
        args.batch_size,
        args.num_workers,
        args.gpus,
        gpu,
        train=False,
    )

    target_mean, target_mad = train_loader.dataset.calc_stats()

    if args.log:
        wandb.init(
            project="SEGNN_QM9_2" + args.dataset,
            name=f'c{args.use_constrained} v2 {args.target} {args.ID}',
            config=args,
            entity=args.wandb_entity,
        )

    def log_metrics(metrics, *, step=None, commit=True):
        if args.log:
            wandb.log(metrics, step=step, commit=commit)

    gammas = model.get_gammas()
    optimizer_dual = None

    no_gammas = len(gammas)
    lambdas = []
    us = []

    if args.resilience == 1:
        us = [
            nn.Parameter(
                torch.zeros(
                    1, dtype=torch.float, requires_grad=True, device=device
                ).squeeze()
            )
            for _ in range(no_gammas)
        ]
        primal_params = list(model.parameters()) + us
    else:
        primal_params = list(model.parameters())

    optimizer = torch.optim.Adam(primal_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.8 * args.epochs), int(0.9 * args.epochs)])
    criterion = nn.L1Loss()

    target = args.target
    best_valid_MAE = 1e30

    if args.disable_equivariance:
        best_valid_MAE_nonequi = 1e30

    if args.use_constrained:
        lambdas = [
            torch.zeros(
                1, dtype=torch.float, requires_grad=False, device=device
            ).squeeze()
            for _ in range(no_gammas)
        ]

        if args.optimizer == 'adam':
            optimizer_dual = torch.optim.Adam(lambdas, lr=args.dual_lr, weight_decay=0)
        elif args.optimizer == 'sgd':
            optimizer_dual = torch.optim.SGD(lambdas, lr=args.dual_lr, weight_decay=0)
        else:
            optimizer_dual = torch.optim.Adam(lambdas, lr=args.dual_lr, weight_decay=0)

    if gpu == 0:
        print("Training:", args.ID)
    for epoch in range(args.epochs):
        epoch_step = global_epoch + epoch
        epoch_metrics = {}
        if args.gpus > 1:
            train_loader.sampler.set_epoch(epoch)
        train_losses = []

        torch.cuda.synchronize()
        start = time.time()
        for step, graph in enumerate(train_loader):
            graph = graph.to(device)
            out = model(graph).squeeze()
            loss = criterion(out, (graph.y - target_mean)/target_mad)

            if args.use_constrained:
                if args.resilience == 1:
                    loss += args.rho/2 * torch.norm(torch.stack(us)) ** 2

                gammas = model.get_gammas()

                if args.resilience == 0:
                    slacks = [
                            gamma - args.epsilon if args.epsilon == 0 else torch.norm(gamma) - args.epsilon
                            for gamma in gammas
                        ]
                else:
                    slacks = [
                            torch.norm(gamma) - c for gamma, c in zip(gammas, us)
                        ]

                dual_loss = 0.0

                for dual_var, slack in zip(lambdas, slacks):
                    dual_loss += dual_var * slack + args.auglag_const * slack ** 2

                if args.resilience:
                    for u, slack in zip(us, slacks):
                        dual_loss -= (u*slack)

                loss = loss + dual_loss

                if args.resilience == 0:
                    for ii, slack in enumerate(slacks):
                        lambdas[ii].grad = None if slack is None else -slack
                else:
                    for ii, (slack, u) in enumerate(zip(slacks, us)):
                        lambdas[ii].grad = None if slack is None else -(slack - u)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.use_constrained:
                optimizer_dual.step()
                if args.epsilon > 0 or args.resilience != 0:
                    for lambda_ in lambdas:
                        lambda_.clamp_(min=0)

            # Track training loss for epoch-level logging.
            train_losses.append(loss.item())

        torch.cuda.synchronize()
        end = time.time()
        print("(TRAIN) epoch:%2d time: %0.4f" %
              (epoch, end - start))

        epoch_metrics["train_loss"] = np.mean(train_losses)

        start = time.time()
        model.set_equivariance(disable_equivariance=False)
        valid_MAE = evaluate(model, valid_loader, criterion, device, args.gpus, target_mean, target_mad)
        end = time.time()
        print("(VALIDATION) epoch:%2d time: %0.4f" %
                (epoch, end - start))

        if args.disable_equivariance:
            model.set_equivariance(disable_equivariance=True)
            valid_MAE_nonequi = evaluate(model, valid_loader, criterion, device, args.gpus, target_mean, target_mad)

        print("VALIDATION: epoch:%2d  step:%4d  %s-MAE(EQ):%0.4f" %
                (epoch, step, target, valid_MAE))
        if args.log:
            epoch_metrics[target + "_val_MAE"] = valid_MAE

        if args.disable_equivariance:
            print("VALIDATION: epoch:%2d  step:%4d  %s-MAE(NEQ):%0.4f" %
                    (epoch, step, target, valid_MAE_nonequi))
            if args.log:
                epoch_metrics[target + "_val_MAE_nonequi"] = valid_MAE_nonequi

            if valid_MAE_nonequi < best_valid_MAE_nonequi:
                best_valid_MAE_nonequi = valid_MAE_nonequi
                utils.save_model(model, args.save_dir, f'{args.ID}_nonequi', args.use_constrained, device)


            if args.use_constrained:
                constraint_metrics = {
                    "gammas/Gamma_" + str(i): gamma.item()
                    for i, gamma in enumerate(gammas)
                }
                constraint_metrics.update({
                    "lambdas/Lambda_" + str(i): lambda_.item()
                    for i, lambda_ in enumerate(lambdas)
                })

                if args.resilience == 1:
                    constraint_metrics.update({
                        "us/U_" + str(i): u.item()
                        for i, u in enumerate(us)
                    })

                epoch_metrics.update(constraint_metrics)

        if valid_MAE < best_valid_MAE:
            best_valid_MAE = valid_MAE
            utils.save_model(model, args.save_dir, args.ID, args.use_constrained, device)

        if args.log:
            log_metrics(epoch_metrics, step=epoch_step)

        scheduler.step()

    model = utils.load_model(model, args.save_dir, args.ID, args.use_constrained, device)
    model.set_equivariance(disable_equivariance=False)

    test_MAE = evaluate(model, test_loader, criterion, device, args.gpus, target_mean, target_mad, )

    if args.disable_equivariance:
        model.set_equivariance(disable_equivariance=True)
        test_MAE_noneq = evaluate(model, test_loader, criterion, device, args.gpus, target_mean, target_mad, )

        model = utils.load_model(model, args.save_dir, f'{args.ID}_nonequi', args.use_constrained, device)
        test_MAE_noneq_by_noneq = evaluate(model, test_loader, criterion, device, args.gpus, target_mean, target_mad, )

    print("TEST: epoch:%2d  step:%4d  %s-MAE:%0.4f" %
            (epoch, step, target, test_MAE))
    test_metrics = {}
    if args.log:
        test_metrics[target + "_test_MAE"] = test_MAE

    if args.disable_equivariance:
        print("TEST: epoch:%2d  step:%4d  %s-MAE(NEQ):%0.4f" %
                (epoch, step, target, test_MAE_noneq))

        print("TEST: epoch:%2d  step:%4d  %s-MAE(NEQ by NEQ):%0.4f" %
                (epoch, step, target, test_MAE_noneq_by_noneq))
        if args.log:
            test_metrics[target + "_test_MAE_nonequi"] = test_MAE_noneq
            test_metrics[target + "_test_MAE_noneq_by_noneq"] = test_MAE_noneq_by_noneq

    if args.log:
        log_metrics(test_metrics, step=global_epoch + args.epochs)

    if args.log and gpu == 0:
        wandb.finish()
    if args.gpus > 1:
        dist.destroy_process_group()

    return best_valid_MAE, test_MAE, 0
