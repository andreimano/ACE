import argparse
from argparse import Namespace
import torch
import torch.utils.data
from motion.dataset import MotionDynamicsDataset as MotionDataset
from model.egno import EGNO
import os
from torch import nn, optim
import json

import random
import numpy as np

from utils import EarlyStopping

from pytorch3d.transforms import Rotate, random_rotations

import wandb

parser = argparse.ArgumentParser(description='EGNO')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='exp_results', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--model', type=str, default='egno', metavar='N')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--delta_frame', type=int, default=30,
                    help='Number of frames delta.')
parser.add_argument('--data_dir', type=str, default='',
                    help='Data directory.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--config_by_file", default=None, nargs="?", const='', type=str, )

parser.add_argument('--lambda_link', type=float, default=1,
                    help='The weight of the linkage loss.')
parser.add_argument('--n_cluster', type=int, default=3,
                    help='The number of clusters.')
parser.add_argument('--flat', action='store_true', default=False,
                    help='flat MLP')
parser.add_argument('--interaction_layer', type=int, default=3,
                    help='The number of interaction layers per block.')
parser.add_argument('--pooling_layer', type=int, default=3,
                    help='The number of pooling layers in EGPN.')
parser.add_argument('--decoder_layer', type=int, default=1,
                    help='The number of decoder layers.')

parser.add_argument('--case', type=str, default='walk',
                    help='The case, walk or run.')

parser.add_argument('--num_timesteps', type=int, default=1,
                    help='The number of time steps.')
parser.add_argument('--time_emb_dim', type=int, default=32,
                    help='The dimension of time embedding.')
parser.add_argument('--num_modes', type=int, default=2,
                    help='The number of modes.')

parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--exp_prefix', type=str, default='constrained', help='Prefix to add before the experiment name for WandB.')
parser.add_argument('--num_runs', type=int, default=5, help='Number of independent runs to execute.')
parser.add_argument('--disable_equi', type=int, default=1, help='If 1, perform dual evaluation: disable equivariance during evaluation')
parser.add_argument('--use_constrained', type=int, default=1, help='If 1, use constrained optimization with primal-dual method')
parser.add_argument('--lr_dual', type=float, default=5e-4, help='Learning rate for the dual optimizer')
parser.add_argument('--auglag_const', type=float, default=0., help='Augmented Lagrangian constant')

parser.add_argument('--use_wandb', type=int, default=1, help='Use wandb or not')
parser.add_argument('--wandb_entity', type=str, default='mls-2', help='Weights & Biases entity')

parser.add_argument('--epsilon', type=float, default=0, help='The epsilon for the constraint')

parser.add_argument('--log_transform_variance', type=int, default=0, help='If 1, log the variance of the embeddings for each sample')

parser.add_argument('--resilience', type=int, default=1, help='If 1, use the resilience constraint')

parser.add_argument('--gamma', type=int, default=1, help='The weight of the resilience constraint')

args = parser.parse_args()
if args.config_by_file is not None:
    if len(args.config_by_file) == 0:
        job_param_path = './configs/config_mocap_no.json'
    else:
        job_param_path = args.config_by_file
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        args = vars(args)
        args.update((k, v) for k, v in hyper_params.items() if k in args)
        args = Namespace(**args)

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss(reduction='none')

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(os.path.join(args.outf, args.exp_name))
except OSError:
    pass

def run_once(run_index, args, start_step=0):
    print(f"\n========== Starting run {run_index+1}/{args.num_runs} ==========")
    seed = args.seed + run_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    dataset_train = MotionDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame, case=args.case, num_timesteps=args.num_timesteps)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=0)

    dataset_val = MotionDataset(partition='val', max_samples=600, data_dir=args.data_dir,
                                delta_frame=args.delta_frame, case=args.case, num_timesteps=args.num_timesteps)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=0)

    dataset_test = MotionDataset(partition='test', max_samples=600, data_dir=args.data_dir,
                                 delta_frame=args.delta_frame, case=args.case, num_timesteps=args.num_timesteps)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=0)

    if args.model == 'egno':
        model = EGNO(n_layers=args.n_layers, in_node_nf=2, in_edge_nf=2, hidden_nf=args.nf, device=device, with_v=True,
                    flat=args.flat, activation=nn.SiLU(), use_time_conv=True, num_modes=args.num_modes,
                    num_timesteps=args.num_timesteps, time_emb_dim=args.time_emb_dim)
    else:
        raise NotImplementedError('Unknown model:', args.model)

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_save_path = os.path.join(args.outf, args.exp_name, f'saved_model_run{run_index}.pth')
    print(f'Model saved to {model_save_path}')
    early_stopping = EarlyStopping(patience=200, verbose=True, path=model_save_path)

    optimizer_dual = None
    lambdas = None
    if args.disable_equi:
        gammas = model.get_gammas()
        lambdas = [
            torch.zeros(1, dtype=torch.float, requires_grad=False, device=device).squeeze() for _ in gammas
        ]

        print(f'Gammas (total: {len(gammas)}):')
        for i, gamma in enumerate(gammas):
            print(f"Gamma {i}: {gamma.item()}")
        print(f'Lambdas (total: {len(lambdas)}):')
        for i, l in enumerate(lambdas):
            print(f"Lambda {i}: {l.item()}")

        if args.use_constrained:
            param_groups = lambdas
            optimizer_dual = optim.Adam(params=param_groups, lr=args.lr_dual, weight_decay=0, eps=1e-4)

    results = {'eval epoch_equivariant': [], 'val loss_equivariant': [], 'test loss_equivariant': [], 'train loss': []}
    if args.disable_equi:
        results.update({'val loss_non_equivariant': [], 'test loss_non_equivariant': []})

    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    best_lp_loss = 1e8

    if args.disable_equi:
        best_val_loss_non_equi = 1e8
        best_test_loss_non_equi = 1e8
        best_val_loss_non_equi_by_non_equi = 1e8
        best_test_loss_non_equi_by_non_equi = 1e8

    for epoch in range(0, args.epochs):
        current_step = start_step + epoch
        train_loss, lp_loss, equi_error = train_epoch(
            model,
            optimizer,
            optimizer_dual,
            lambdas,
            epoch,
            loader_train,
            args,
            constrained=args.use_constrained,
            disable_equivariance=args.disable_equi,
            current_step=current_step,
        )

        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss_equi, _ = test_epoch(model, epoch, loader_val, args)
            test_loss_equi, _ = test_epoch(model, epoch, loader_test, args)

            if args.disable_equi:
                val_loss_non_equi, _ = test_epoch(model, epoch, loader_val, args, disable_equivariance=True)
                test_loss_non_equi, _ = test_epoch(model, epoch, loader_test, args, disable_equivariance=True)

            results['eval epoch_equivariant'].append(epoch)
            results['val loss_equivariant'].append(val_loss_equi)
            results['test loss_equivariant'].append(test_loss_equi)

            if args.disable_equi:
                results['val loss_non_equivariant'].append(val_loss_non_equi)
                results['test loss_non_equivariant'].append(test_loss_non_equi)

            if val_loss_equi < best_val_loss:
                best_val_loss = val_loss_equi
                best_test_loss = test_loss_equi
                best_train_loss = train_loss
                best_epoch = epoch
                best_lp_loss = lp_loss

                best_val_loss_non_equi = val_loss_non_equi if args.disable_equi else None
                best_test_loss_non_equi = test_loss_non_equi if args.disable_equi else None

                if args.disable_equi:
                    early_stopping(val_loss_non_equi, model)
                else:
                    early_stopping(val_loss_equi, model)

                if early_stopping.early_stop:
                    print("Early Stopping.")
                    break

            if args.disable_equi:
                if val_loss_non_equi < best_val_loss_non_equi_by_non_equi:
                    best_val_loss_non_equi_by_non_equi = val_loss_non_equi
                    best_test_loss_non_equi_by_non_equi = test_loss_non_equi

            if args.disable_equi:
                print(f"*** [Run {run_index}] Best Val Loss (Equivariant): {best_val_loss:.5f} \t Best Test Loss (Equivariant): {best_test_loss:.5f} \t Best Val Loss (Non-Equivariant, equi-based): {best_val_loss_non_equi:.5f} \t Best Test Loss (Non-Equivariant, equi-based): {best_test_loss_non_equi:.5f} \t Best Val Loss (Non-Equivariant, non-equi based): {best_val_loss_non_equi_by_non_equi:.5f} \t Best Test Loss (Non-Equivariant, non-equi based): {best_test_loss_non_equi_by_non_equi:.5f} \t Best epoch {best_epoch}")
            else:
                print(f"*** [Run {run_index}] Best Val Loss: {best_val_loss:.5f} \t Best Test Loss: {best_test_loss:.5f} \t Best epoch {best_epoch}")

            if args.disable_equi:
                gammas = model.get_gammas()
                lambda_values = [l.item() for l in lambdas]
                gamma_values = [g.item() for g in gammas]

                for idx in range(len(lambdas)):
                    wandb.log({
                        f"lambdas/lambda_{idx}": lambda_values[idx],
                        f"gammas/gamma_{idx}": gamma_values[idx],
                    }, step=current_step, commit=False)

            wandb.log({
                "train_loss": train_loss,
                "val_loss_equi": val_loss_equi,
                "test_loss_equi": test_loss_equi,
                "val_loss_non_equi": val_loss_non_equi if args.disable_equi else None,
                "test_loss_non_equi": test_loss_non_equi if args.disable_equi else None,
                "equi_error/mean": equi_error,
            }, step=current_step, commit=True)

            json_object = json.dumps(results, indent=4)
            loss_file_path = os.path.join(args.outf, args.exp_name, f"loss_run{run_index}.json")
            with open(loss_file_path, "w") as outfile:
                outfile.write(json_object)

    final_step = start_step + args.epochs
    wandb.log({
        "final/final_best_train_loss_run": best_train_loss,
        "final/final_best_val_loss_equi_run": best_val_loss,
        "final/final_best_test_loss_equi_run": best_test_loss,
        "final/final_best_epoch_run": best_epoch,
        "final/final_best_lp_loss_run": best_lp_loss,
        "final/final_best_val_loss_non_equi_run": best_val_loss_non_equi if args.disable_equi else None,
        "final/final_best_test_loss_non_equi_run": best_test_loss_non_equi if args.disable_equi else None,
        "final/final_best_val_loss_non_equi_by_non_equi_run": best_val_loss_non_equi_by_non_equi if args.disable_equi else None,
        "final/final_best_test_loss_non_equi_by_non_equi_run": best_test_loss_non_equi_by_non_equi if args.disable_equi else None,
    }, step=final_step, commit=True)

    if args.disable_equi:
        return best_train_loss, best_val_loss, best_test_loss, best_epoch, best_lp_loss, final_step + 1, best_val_loss_non_equi, best_test_loss_non_equi, best_val_loss_non_equi_by_non_equi, best_test_loss_non_equi_by_non_equi
    else:
        return best_train_loss, best_val_loss, best_test_loss, best_epoch, best_lp_loss, final_step + 1, None, None, None, None

def train_epoch(
    model,
    optimizer,
    optimizer_dual,
    lambdas,
    epoch,
    loader,
    args,
    constrained=False,
    disable_equivariance=False,
    current_step=0,
):
    model.train()
    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'lp_loss': 0}
    equi_errors = []

    use_constraints = constrained and lambdas is not None and len(lambdas) > 0
    cs = torch.stack(lambdas) if use_constraints else None

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        for i in [-1, -2]:
            d = data[i].view(batch_size * n_nodes, args.num_timesteps, 3)
            data[i] = d.transpose(0, 1).contiguous().view(-1, 3)

        loc, vel, edges, edge_attr, _, _, Z, loc_end, vel_end = data
        loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, loc.size(2))
        loc = loc.view(-1, loc.size(2))
        vel = vel.view(-1, vel.size(2))
        offset = (torch.arange(batch_size) * n_nodes).unsqueeze(-1).unsqueeze(-1).to(edges.device)
        edges = torch.cat(list(edges + offset), dim=-1)
        edge_attr = torch.cat(list(edge_attr), dim=0)
        Z = Z.view(-1, Z.size(2))

        optimizer.zero_grad()

        edge_attr_init = edge_attr

        if args.model == 'egno':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
            loc_pred, vel_pred, _ = model(
                loc,
                nodes,
                edges,
                edge_attr,
                v=vel,
                loc_mean=loc_mean,
                disable_equivariance=disable_equivariance,
            )
        else:
            raise Exception("Wrong model")

        losses = loss_mse(loc_pred, loc_end).view(args.num_timesteps, batch_size * n_nodes, 3)
        losses = torch.mean(losses, dim=(1, 2))

        if args.resilience == 1 and cs is not None:
            loss = torch.mean(losses) + args.gamma/2 * torch.norm(cs)**2
        else:
            loss = torch.mean(losses)

        if use_constraints:
            gammas = model.get_gammas()
            if args.epsilon == 0:
                slacks = [gamma for gamma in gammas]
            elif args.resilience == 1:
                slacks = [torch.norm(gamma) - c for gamma, c in zip(gammas, cs)]
            elif args.resilience == 0 and args.epsilon > 0:
                slacks = [torch.norm(gamma) - args.epsilon for gamma in gammas]
            else:
                raise Exception("Constraint not implemented")

            dual_loss = 0

            for dual_var, slack in zip(lambdas, slacks):
                dual_loss += dual_var * slack + args.auglag_const * slack ** 2

            loss += dual_loss

            for ii, slack in enumerate(slacks):
                lambdas[ii].grad = None if slack is None else -slack

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if use_constraints:
            optimizer_dual.step()
            if args.epsilon > 0 or args.resilience != 0:
                for lambda_ in lambdas:
                    lambda_.clamp_(min=0)
                    
        res['loss'] += losses[-1].item() * batch_size
        res['counter'] += batch_size
  

        if hasattr(args, 'log_transform_variance') and args.log_transform_variance:

            with torch.no_grad():
                model.eval()
                loc_pred, vel_pred, _ = model(
                    loc,
                    nodes,
                    edges,
                    edge_attr,
                    v=vel,
                    loc_mean=loc_mean,
                    disable_equivariance=disable_equivariance,
                )      

                loc, _, _, _, _, _, _, _, _ = data

                n_transforms = 3
                for _ in range(n_transforms):
                    R = Rotate(R=random_rotations(batch_size, device=loc.device))
                    loc_batched = loc.reshape(batch_size, n_nodes, 3)
                    loc_rot_batched = R.transform_points(loc_batched)
                    loc_mean_rotated = loc_rot_batched.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, loc.size(2))
                    loc_rot = loc_rot_batched.reshape(-1, 3)

                    vel_batched = vel.reshape(batch_size, n_nodes, 3)
                    vel_rot_batched = R.transform_points(vel_batched)
                    vel_rot = vel_rot_batched.reshape(-1, 3)

                    loc_dist_rot = torch.sum((loc_rot[rows] - loc_rot[cols])**2, 1).unsqueeze(1)
                    edge_attr_rot = torch.cat([edge_attr_init, loc_dist_rot], 1).detach()

                    nodes_rot = torch.sqrt(torch.sum(vel_rot ** 2, dim=1)).unsqueeze(1).detach()
                    nodes_rot = torch.cat((nodes_rot, Z / Z.max()), dim=-1)

                    loc_pred_rot, vel_pred_rot, _ = model(
                        loc_rot,
                        nodes_rot,
                        edges,
                        edge_attr_rot,
                        v=vel_rot,
                        loc_mean=loc_mean_rotated,
                        disable_equivariance=disable_equivariance,
                    )

                    loc_pred_rot_last = loc_pred_rot.view(args.num_timesteps, batch_size, n_nodes, 3)[-1]
                    loc_pred_last = loc_pred.view(args.num_timesteps, batch_size, n_nodes, 3)[-1]

                    loc_pred_init_rot_last = R.transform_points(loc_pred_last)
                    diff_equi = (loc_pred_rot_last - loc_pred_init_rot_last).norm(dim=-1).mean()
                    equi_errors.append(diff_equi)

            model.train()

    prefix = ""
    print(f"{prefix} epoch {epoch} avg loss: {res['loss'] / res['counter']:.5f} avg lploss: {res['lp_loss'] / res['counter']:.5f}")
    equi_error = torch.mean(torch.stack(equi_errors)).item() if equi_errors else 0.0
    return res['loss'] / res['counter'], res['lp_loss'] / res['counter'], equi_error

def test_epoch(model, epoch, loader, args, disable_equivariance=False):
    model.eval()
    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'lp_loss': 0}
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            batch_size, n_nodes, _ = data[0].size()
            data = [d.to(device) for d in data]
            for i in [-1, -2]:
                d = data[i].view(batch_size * n_nodes, args.num_timesteps, 3)
                data[i] = d.transpose(0, 1).contiguous().view(-1, 3)

            loc, vel, edges, edge_attr, _, _, Z, loc_end, vel_end = data
            loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, loc.size(2))
            loc = loc.view(-1, loc.size(2))
            vel = vel.view(-1, vel.size(2))
            offset = (torch.arange(batch_size) * n_nodes).unsqueeze(-1).unsqueeze(-1).to(edges.device)
            edges = torch.cat(list(edges + offset), dim=-1)
            edge_attr = torch.cat(list(edge_attr), dim=0)
            Z = Z.view(-1, Z.size(2))

            if args.model == 'egno':
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
                nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
                rows, cols = edges
                loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)
                edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
                loc_pred, vel_pred, _ = model(
                    loc,
                    nodes,
                    edges,
                    edge_attr,
                    v=vel,
                    loc_mean=loc_mean,
                    disable_equivariance=disable_equivariance,
                )
            else:
                raise Exception("Wrong model")

            losses = loss_mse(loc_pred, loc_end).view(args.num_timesteps, batch_size * n_nodes, 3)
            losses = torch.mean(losses, dim=(1, 2))

            res['loss'] += losses[-1].item() * batch_size
            res['counter'] += batch_size

    prefix = "==> "
    print(f"{prefix} epoch {epoch} avg loss: {res['loss'] / res['counter']:.5f} avg lploss: {res['lp_loss'] / res['counter']:.5f}")
    return res['loss'] / res['counter'], res['lp_loss'] / res['counter']

def main():
    global_run_name = f"eps{args.epsilon}_ds{args.case}_{args.exp_prefix}{args.exp_name}_{args.model}_lr{args.lr}_nf{args.nf}_epochs{args.epochs}_numruns{args.num_runs}"
    wandb.init(project="EGNO_project", entity=args.wandb_entity, name=global_run_name, config=vars(args), mode="online" if args.use_wandb else "disabled")

    final_test_metrics = []
    final_train_metrics = []
    final_val_metrics = []
    final_val_metrics_nonequi = []
    final_test_metrics_nonequi = []
    final_lp_metrics = []
    best_epochs = []
    final_val_metrics_nonequi_by_non_equi = []
    final_test_metrics_nonequi_by_non_equi = []

    current_step_offset = 0

    for run in range(args.num_runs):
        (
            best_train,
            best_val,
            best_test,
            best_epoch,
            best_lp,
            new_offset,
            best_val_non_equi,
            best_test_non_equi,
            best_val_non_equi_by_non_equi,
            best_test_non_equi_by_non_equi,
        ) = run_once(run, args, start_step=current_step_offset)

        final_test_metrics.append(best_test)
        final_val_metrics.append(best_val)
        final_train_metrics.append(best_train)
        final_lp_metrics.append(best_lp)
        best_epochs.append(best_epoch)

        if args.disable_equi:
            final_val_metrics_nonequi.append(best_val_non_equi)
            final_test_metrics_nonequi.append(best_test_non_equi)
            final_val_metrics_nonequi_by_non_equi.append(best_val_non_equi_by_non_equi)
            final_test_metrics_nonequi_by_non_equi.append(best_test_non_equi_by_non_equi)

        print(f"Run {run} finished with best test loss: {best_test:.6f}")
        current_step_offset = new_offset

    final_test_metrics = np.array(final_test_metrics)
    mean_metric = final_test_metrics.mean()
    std_metric = final_test_metrics.std()

    print("========================================")
    print(f"Final test metrics over {args.num_runs} runs: {final_test_metrics}")
    print(f"Mean: {mean_metric:.6f}, Std: {std_metric:.6f}")
    print("========================================")

    wandb.log({
        "final/final_test_metric_mean": mean_metric,
        "final/final_test_metric_std": std_metric,
        
        "final/final_val_metric_mean": np.mean(final_val_metrics),
        "final/final_val_metric_std": np.std(final_val_metrics),

        "final/final_test_metric_nonequi_by_non_equi_mean": np.mean(final_test_metrics_nonequi_by_non_equi) if args.disable_equi else None,
        "final/final_val_metric_nonequi_by_non_equi_mean": np.mean(final_val_metrics_nonequi_by_non_equi) if args.disable_equi else None,
        "final/final_test_metric_nonequi_by_non_equi_std": np.std(final_test_metrics_nonequi_by_non_equi) if args.disable_equi else None,
        "final/final_val_metric_nonequi_by_non_equi_std": np.std(final_val_metrics_nonequi_by_non_equi) if args.disable_equi else None,
    }, step=current_step_offset, commit=True)

    wandb.finish()

if __name__ == "__main__":
    main()
