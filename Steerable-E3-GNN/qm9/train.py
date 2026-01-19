import wandb
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR

from qm9.dataset import QM9
from qm9.evaluate import evaluate
import utils
import time


def train(gpu, model, args, global_epoch):
    if args.gpus == 0:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
        if args.gpus > 1:
            dist.init_process_group("nccl", rank=gpu, world_size=args.gpus)
            torch.cuda.set_device(gpu)

    model = model.to(device)

    # count params
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    model = torch.compile(model)
    # if args.gpus > 1:
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    # Create datasets and dataloaders
    train_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "train", args.lmax_attr,
                                             feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu)
    valid_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "valid", args.lmax_attr,
                                             feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu, train=False)
    test_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "test", args.lmax_attr,
                                            feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu, train=False)

    # Get train set statistics
    target_mean, target_mad = train_loader.dataset.calc_stats()


    # Init wandb
    if args.log:
        wandb.init(project="SEGNN_QM9_2" + args.dataset, name=f'c{args.use_constrained} v2 {args.target} {args.ID}', config=args, entity="mls-2")

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


    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(primal_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.8*(args.epochs)), int(0.9*(args.epochs))], verbose=True)
    criterion = nn.L1Loss()

    # Logging parameters
    target = args.target
    best_valid_MAE = 1e30
    
    if args.disable_equivariance:
        best_valid_MAE_nonequi = 1e30

    i = 0
    N_samples = 0
    loss_sum = 0
    train_MAE_sum = 0

    if args.use_constrained:
        no_gammas = len(gammas)

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


    # Let's start!
    if gpu == 0:
        print("Training:", args.ID)
    for epoch in range(args.epochs):
        # Set epoch so shuffling works right in distributed mode.
        if args.gpus > 1:
            train_loader.sampler.set_epoch(epoch)
        # Training loop

        # if args.use_constrained and args.resilience == 1:
        #     us = torch.stack(lambdas)

        train_losses = []

        torch.cuda.synchronize()
        start = time.time()
        for step, graph in enumerate(train_loader):
            # Forward & Backward.
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

            # Logging
            i += 1
            N_samples += graph.y.size(0)
            loss_sum += loss
            train_MAE_sum += criterion(out.detach()*target_mad + target_mean, graph.y)*graph.y.size(0)

            train_losses.append(loss.item())

            # Report
            if i % args.print == 0 and False:
                print("epoch:%2d  step:%4d  loss: %0.4f  train MAE:%0.4f" %
                      (epoch, step, loss_sum/i, train_MAE_sum/N_samples))

                # if args.log and gpu == 0:
                #     wandb.log({"loss": loss_sum/i, target + " train MAE": train_MAE_sum /
                #                N_samples})

                i = 0
                N_samples = 0
                loss_sum = 0
                train_MAE_sum = 0

        torch.cuda.synchronize()
        end = time.time()
        print("(TRAIN) epoch:%2d time: %0.4f" %
              (epoch, end - start))
        

        wandb.log({"train_loss": np.mean(train_losses)}, step=epoch)

        # Evaluate on validation set
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
            wandb.log({target + "_val_MAE": valid_MAE}, step=epoch)

        if args.disable_equivariance:
            print("VALIDATION: epoch:%2d  step:%4d  %s-MAE(NEQ):%0.4f" %
                    (epoch, step, target, valid_MAE_nonequi))
            if args.log:
                wandb.log({target + "_val_MAE_nonequi": valid_MAE_nonequi}, step=epoch)

            if valid_MAE_nonequi < best_valid_MAE_nonequi:
                best_valid_MAE_nonequi = valid_MAE_nonequi
                utils.save_model(model, args.save_dir, f'{args.ID}_nonequi', args.use_constrained, device)


            if args.use_constrained:
                #log lambdas and gammas to wandb
                for i, gamma in enumerate(gammas):
                    wandb.log({"gammas/Gamma_" + str(i): gamma.item()}, commit=False, step = global_epoch + epoch)
                
                for i, lambda_ in enumerate(lambdas):
                    wandb.log({"lambdas/Lambda_" + str(i): lambda_.item()}, commit=False, step = global_epoch + epoch)

                if args.resilience == 1:
                    for i, u in enumerate(us):
                        wandb.log({"us/U_" + str(i): u.item()}, commit=False, step = global_epoch + epoch)

        if valid_MAE < best_valid_MAE:
            best_valid_MAE = valid_MAE
            utils.save_model(model, args.save_dir, args.ID, args.use_constrained, device)

        # Adapt learning rate
        scheduler.step()

    # Final evaluation on test set
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
    if args.log:
        wandb.log({target + "_test_MAE": test_MAE})


    if args.disable_equivariance:
        print("TEST: epoch:%2d  step:%4d  %s-MAE(NEQ):%0.4f" %
                (epoch, step, target, test_MAE_noneq))
        
        print("TEST: epoch:%2d  step:%4d  %s-MAE(NEQ by NEQ):%0.4f" %
                (epoch, step, target, test_MAE_noneq_by_noneq))
        if args.log:
            wandb.log({target + "_test_MAE_nonequi": test_MAE_noneq})
            wandb.log({target + "_test_MAE_noneq_by_noneq": test_MAE_noneq_by_noneq})

    if args.log and gpu == 0:
        wandb.finish()
    if args.gpus > 1:
        dist.destroy_process_group()

    # return best val loss, best test loss, best epoch
    # if args.disable_equivariance:
    #     return (best_valid_MAE, best_valid_MAE_nonequi), (test_MAE, test_MAE_noneq, test_MAE_noneq_by_noneq), 0
    # else:
    return best_valid_MAE, test_MAE, 0