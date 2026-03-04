"""
Author: Congyue Deng
Contact: congyue@stanford.edu
Date: April 2021
"""

import argparse
import importlib
import logging
import os
import sys

import numpy as np
import provider
import torch
from data_utils.ModelNetDataLoader import ModelNetDataLoader, ModelNet40
from pytorch3d.transforms import Rotate, RotateAxisAngle, random_rotations
from tqdm import tqdm
from utils.utils import get_optimizer, get_scheduler, setup_dir_and_logging
import time

import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

def parse_args():
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument("--model", default="vn_dgcnn_cls", help="Model name", choices=["pointnet_cls", "vn_pointnet_cls", "dgcnn_cls", "vn_dgcnn_cls"])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size in training")
    parser.add_argument("--epoch", default=250, type=int, help="Number of epochs in training")

    parser.add_argument("--lr", default=1e-1, type=float, help="Initial learning rate (for SGD it is multiplied by 100)")
    parser.add_argument("--dual_lr", type=float, default=1e-1, help="Learning rate for dual variables")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer for training")

    parser.add_argument("--scheduler_primal", type=str, default='None', help="Scheduler type for primal optimizer", choices=['cosine', 'step', 'None'])
    parser.add_argument("--scheduler_dual", type=str, default='None', help="Scheduler type for dual optimizer", choices=['cosine', 'step', 'None'])

    parser.add_argument("--num_point", type=int, default=1024, help="Number of points")
    parser.add_argument("--log_dir", type=str, default="vn_dgcnn/aligned", help="Experiment root")
    parser.add_argument("--normal", action="store_true", default=False, help="Use normal information")
    parser.add_argument("--rot", type=str, default="so3", help="Rotation augmentation to input data", choices=["aligned", "z", "so3"])
    parser.add_argument("--pooling", type=str, default="mean", help="VNN pooling method", choices=["mean", "max"])
    parser.add_argument("--n_knn", default=20, type=int, help="Number of nearest neighbors")
    parser.add_argument("--continue_training", action="store_true", default=False)
    parser.add_argument("--use_wandb", type=int, default=1, help="Use wandb for logging")
    parser.add_argument("--wandb_entity", type=str, default='mls-2', help="Wandb entity")

    parser.add_argument("--reparametrize", type=str, default="None", help="Reparametrize the linear layers")

    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon value for constrained optimization")
    parser.add_argument("--dual_warmup", type=int, default=3, help="Dual warmup factor")
    parser.add_argument("--primal_warmup", type=int, default=3, help="Primal warmup factor")

    parser.add_argument("--use_constrained", type=int, default=1, help="Use constrained optimization")
    parser.add_argument("--disable_equivariance", type=int, default=1, help="Disable equivariance")

    parser.add_argument("--wandb_prefix", type=str, default="", help="Wandb experiment prefix")

    parser.add_argument("--watch_model", type=int, default=0, help="Watch model with wandb")

    parser.add_argument("--wd", type=float, default=0.0, help="Decay rate")
    parser.add_argument("--dual_wd", type=float, default=0.0, help="Weight decay for linear skip layer")

    parser.add_argument("--gamma_init_val", "--theta_init_val", dest="gamma_init_val", type=float, default=1., help="Gamma init value")

    parser.add_argument("--data_path", type=str, default="./data/modelnet40_normal_resampled/", help="Path to data")

    parser.add_argument("--always_rotate_test", type=int, default=1, help="Always rotate test data")

    parser.add_argument("--h5_dataset", type=int, default=0, help="Use h5 dataset")

    parser.add_argument("--extra_augments", type=int, default=0, help="Use extra data augmentations")

    parser.add_argument("--point_drop", type=int, default=0, help="Use point dropout")

    parser.add_argument("--full_nonequi", type=int, default=1, help="Use full non-equivariant layers")

    parser.add_argument("--dataset_dropout", type=int, default=0, help="Use dataset dropout")

    parser.add_argument("--seed", type=int, default=60, help="Random seed")

    return parser.parse_args()


def test(model, loader, criterion, args, device, num_class=40, disable_equivariance=True):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    losses = []
    classifier = model.eval()
    for _, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data

        trot = None
        if args.rot == "z":
            trot = RotateAxisAngle(
                angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True
            )
        elif args.rot == "so3" or args.always_rotate_test:
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points)

        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        pred, trans_feat = classifier(points, disable_equivariance=disable_equivariance)
        loss = criterion(pred, target.long(), trans_feat)
        losses.append(loss.item())

        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = (
                pred_choice[target == cat]
                .eq(target[target == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0]
            )
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc, np.mean(losses)


def evaluate_and_log(
    classifier,
    testDataLoader,
    criterion,
    best_instance_acc,
    best_class_acc,
    log_string,
    args,
    device,
    disable_equivariance=True,
):
    with torch.no_grad():

        instance_acc, class_acc, inference_avg_loss = test(
            classifier.eval(),
            testDataLoader,
            criterion=criterion,
            args=args,
            device=device,
            disable_equivariance=disable_equivariance,
        )

        improved = instance_acc >= best_instance_acc
        if improved:
            best_instance_acc = instance_acc

        if class_acc >= best_class_acc:
            best_class_acc = class_acc

        suffix = "_equivariant" if not disable_equivariance else "_non_equivariant"

        log_string(
            f"Test Instance Accuracy{suffix}: {instance_acc}, Class Accuracy{suffix}: {class_acc}"
        )

        if improved:
            log_string(
                f"Best Instance Accuracy{suffix}: {best_instance_acc}, Class Accuracy{suffix}: {best_class_acc}"
            )

        test_metrics_dict = {
            f"test_instance_accuracy{suffix}": instance_acc,
            f"test_class_accuracy{suffix}": class_acc,
            f"best_instance_accuracy{suffix}": best_instance_acc,
            f"best_class_accuracy{suffix}": best_class_acc,
            f"inference_avg_loss{suffix}": inference_avg_loss,
        }

    return best_instance_acc, best_class_acc, test_metrics_dict


def main(args):
    logger, experiment_dir, _ = setup_dir_and_logging(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def log_string(str):
        logger.info(str)

    wandb.init(
        entity=args.wandb_entity,
        mode="online" if args.use_wandb else "disabled",
        project=f"dgcnn-ours-{args.num_point}",
        config=vars(args),
    )

    args = argparse.Namespace(**dict(wandb.config))

    if not args.disable_equivariance:
        args.use_constrained = 0

    wandb.config.update(args, allow_val_change=True)

    experiment_name = f"{args.optimizer}_repr_{args.reparametrize}_lrs{args.lr};{args.dual_lr}_wds{args.wd};{args.dual_wd}_{args.model}_opt_{args.optimizer}_eps_{args.epsilon}"

    if args.use_constrained:
        experiment_name = f"CONSTR_{experiment_name}"
    if args.disable_equivariance:
        experiment_name = f"NOEQ_{experiment_name}"

    experiment_name = f"{args.wandb_prefix}{experiment_name}"

    wandb.run.name = experiment_name
    wandb.run.save()

    log_string("Load dataset ...")
    DATA_PATH = args.data_path

    if args.h5_dataset:
        trainDataLoader = torch.utils.data.DataLoader(ModelNet40(partition='train', num_points=args.num_point, base_dir=args.data_path), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(ModelNet40(partition='test', num_points=args.num_point, base_dir=args.data_path), num_workers=8,
                                batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        TRAIN_DATASET = ModelNetDataLoader(
            root=DATA_PATH, npoint=args.num_point, split="train", normal_channel=args.normal, enable_random_dropout=args.dataset_dropout
        )
        TEST_DATASET = ModelNetDataLoader(
            root=DATA_PATH, npoint=args.num_point, split="test", normal_channel=args.normal, enable_random_dropout=args.dataset_dropout
        )
        trainDataLoader = torch.utils.data.DataLoader(
            TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        testDataLoader = torch.utils.data.DataLoader(
            TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

    num_class = 40
    MODEL = importlib.import_module(args.model)

    classifier = MODEL.get_model(args, num_class, normal_channel=args.normal).to(device)
    criterion = MODEL.get_loss().to(device)

    if args.continue_training:
        try:
            checkpoint = torch.load(str(experiment_dir) + "/checkpoints/best_model.pth")
            start_epoch = checkpoint["epoch"]
            classifier.load_state_dict(checkpoint["model_state_dict"])
            log_string("Use pretrain model")
        except:
            log_string("No existing model, starting training from scratch...")
            start_epoch = 0
    else:
        start_epoch = 0

    n_batches = len(trainDataLoader)
    total_steps = n_batches * args.epoch

    skip_params = []

    for name, module in classifier.named_modules():
        if 'dense_lin' in name:
            skip_params.extend(module.parameters())

    other_params = [param for name, param in classifier.named_parameters() if 'dense_lin' not in name]

    params = [
        {'params': skip_params, 'weight_decay': args.wd if args.dual_wd == args.wd else args.dual_wd},
        {'params': other_params, 'weight_decay': args.wd}
    ]

    optimizer_primal = get_optimizer(args, params, lr=args.lr, momentum=0.)
    num_warmup_steps_primal = total_steps // args.primal_warmup if total_steps else 0
    num_warmup_steps_dual = total_steps // args.dual_warmup if total_steps else 0

    scheduler = get_scheduler(
        args.scheduler_primal,
        optimizer_primal,
        total_steps=total_steps,
        num_warmup_steps=num_warmup_steps_primal,
        problem='primal',
        args=args
    )

    if args.use_constrained:
        gammas = classifier.get_gammas()
        no_gammas = len(gammas)

        lambdas = [
            torch.zeros(
                1, dtype=torch.float, requires_grad=False, device=device
            ).squeeze()
            for _ in range(no_gammas)
        ]

        optimizer_dual = get_optimizer(args, lambdas, lr=args.dual_lr)
        scheduler_dual = get_scheduler(
            args.scheduler_dual,
            optimizer_dual,
            total_steps=total_steps,
            num_warmup_steps=num_warmup_steps_dual,
            problem='dual',
            args=args
        )

    global_epoch = 0

    best_instance_acc_equiv = 0.0
    best_class_acc_equiv = 0.0
    best_instance_acc_noneq = 0.0
    best_class_acc_noneq = 0.0

    logger.info("Start training...")

    if args.watch_model and args.use_wandb:
        wandb.watch(classifier, log="all")

    for epoch in range(start_epoch, args.epoch):
        scheduler.step()
        log_string(f"Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):")

        mean_correct = []
        epoch_loss = 0.0
        classifier = classifier.train()
        
        synchronize()
        start = time.time()

        for batch_id, data in tqdm(
            enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
        ):            
            points, target = data

            trot = None
            if args.rot == "z":
                trot = RotateAxisAngle(
                    angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True
                )
            elif args.rot == "so3":
                trot = Rotate(R=random_rotations(points.shape[0]))
            if trot is not None:
                points = trot.transform_points(points)

            if args.extra_augments and args.point_drop:
                points = points.data.numpy()
                points = provider.random_point_dropout(points)
                points = torch.Tensor(points)                

            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            optimizer_primal.zero_grad()

            pred, trans_feat = classifier(points, disable_equivariance=args.disable_equivariance)

            if args.use_constrained:
                slacks = [
                    gamma - args.epsilon if args.epsilon == 0 else torch.norm(gamma) - args.epsilon
                    for gamma in gammas
                ]

            lagrangian = criterion(pred, target.long(), trans_feat)
            loss_val = lagrangian.item()

            if args.use_constrained:
                scheduler_dual.step()
                
                dual_loss = 0.0
                for dual_var, slack in zip(lambdas, slacks):
                    dual_loss += dual_var * slack

                dual_loss /= no_gammas

                lagrangian += dual_loss
                
                for ii, slack in enumerate(slacks):
                    lambdas[ii].grad = None if slack is None else -slack

            lagrangian.backward()
            optimizer_primal.step()

            if args.use_constrained:
                optimizer_dual.step()
                scheduler_dual.step()

            if args.reparametrize == "manual" and args.disable_equivariance:
                with torch.no_grad():
                    non_equi_layers = classifier.get_nonequi_linear_layers()
                    for layer in non_equi_layers:
                        layer.weight.div_(torch.norm(layer.weight))

            epoch_loss += loss_val

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
        synchronize()
        end = time.time()
        log_string(f"Time taken for epoch {epoch + 1}: {end - start:.2f} seconds")

        train_instance_acc = np.mean(mean_correct)
        mean_loss = epoch_loss / len(trainDataLoader)
        log_string(f"Train Instance Accuracy: {train_instance_acc}")
        log_string(f"Train Loss: {mean_loss}")

        classifier.eval()

        start = time.time()
        best_instance_acc_equiv, best_class_acc_equiv, test_metrics_dict_equiv = evaluate_and_log(
            classifier,
            testDataLoader,
            criterion,
            best_instance_acc_equiv,
            best_class_acc_equiv,
            log_string,
            args,
            device,
            disable_equivariance=False,
        )
        synchronize()
        end = time.time()
        log_string(f"Time taken for evaluation (equivariant): {end - start:.2f} seconds")

        test_metrics_dict_noneq = {}
        if args.use_constrained or args.disable_equivariance:
            best_instance_acc_noneq, best_class_acc_noneq, test_metrics_dict_noneq = evaluate_and_log(
                classifier,
                testDataLoader,
                criterion,
                best_instance_acc_noneq,
                best_class_acc_noneq,
                log_string,
                args,
                device,
                disable_equivariance=True,
            )

        if global_epoch == 0:
            num_params = sum(
                p.numel()
                for p in classifier.parameters()
            )
            log_string(f"Number of parameters: {num_params}")
            wandb.log({"num_params": num_params}, step=global_epoch, commit=False)

        lr_dict = {
            "lr": optimizer_primal.param_groups[0]["lr"],
            "dual_lr": (
                optimizer_dual.param_groups[0]["lr"] if args.use_constrained else 0
            ),
        }

        slacks_dict = {}
        if args.use_constrained:
            slacks_dict = {
                f"slacks/{ii}": slack.item() for ii, slack in enumerate(slacks)
            }

        lambdas_dict = {}
        if args.use_constrained:
            lambdas_dict = {
                f"lambdas/{ii}": lambdas[ii].item() for ii in range(no_gammas)
            }

        lambda_grads_dict = {}
        if args.use_constrained:
            lambda_grads_dict = {
                f"lambda_grads/{ii}": lambdas[ii].grad.item() for ii in range(no_gammas)
            }
        
        gammas_dict = {}
        gammas_ = classifier.get_gammas()
        if args.use_constrained:
            gammas_dict = {
                f"gammas/{ii}": gammas_[ii].item() for ii in range(no_gammas)
            }


        train_dict = {
            "train_loss": mean_loss,
            "train_accuracy": train_instance_acc,
        }

        metrics_to_log = {
            **lr_dict,
            **train_dict,
            **test_metrics_dict_equiv,
            **test_metrics_dict_noneq,
            **slacks_dict,
            **lambdas_dict,
            **lambda_grads_dict,
            **gammas_dict
        }

        wandb.log(
            metrics_to_log,
            step=global_epoch,
            commit=True,
        )

        global_epoch += 1

    logger.info("End of training...")


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)
