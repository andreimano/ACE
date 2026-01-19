import argparse
import time

import numpy as np
import torch
import wandb
from e3nn.o3 import Irreps, spherical_harmonics
from pytorch3d.transforms import Rotate, random_rotations
from torch import nn, optim
from torch_geometric.data import Data
from torch_scatter import scatter

from nbody.dataset_nbody import NBodyDataset

time_exp_dic = {'time': 0, 'counter': 0}


class O3Transform:
    def __init__(self, lmax_attr):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)

    def __call__(self, graph):
        pos = graph.pos
        vel = graph.vel
        charges = graph.charges

        prod_charges = charges[graph.edge_index[0]] * charges[graph.edge_index[1]]
        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        graph.edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='integral')
        vel_embedding = spherical_harmonics(self.attr_irreps, vel, normalize=True, normalization='integral')
        graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean") + vel_embedding

        vel_abs = torch.sqrt(vel.pow(2).sum(1, keepdims=True))
        mean_pos = 0

        graph.x = torch.cat((pos - mean_pos, vel, vel_abs), 1)
        graph.additional_message_features = torch.cat((edge_dist, prod_charges), dim=-1)
        return graph

class AdamSGD:
    def __init__(self, model_params, gammas, lr, weight_decay):
        self.adam_opt = optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        self.gamma_opt = optim.SGD(gammas, lr=lr * 1000, weight_decay=0)

    def zero_grad(self):
        self.adam_opt.zero_grad()
        self.gamma_opt.zero_grad()
    
    def step(self):
        self.adam_opt.step()
        self.gamma_opt.step()

def prepare_graph(pos, vel, charges, loc_end, edge_index, batch_index, device, transform):
    """Attach attributes, move data to the proper device, and run the O3 transform."""
    graph = Data(edge_index=edge_index, pos=pos, vel=vel, charges=charges, y=loc_end)
    graph.batch = batch_index.clone()
    graph = graph.to(device)
    return transform(graph)


def compute_slacks(gammas, args, us):
    if args.resilience == 0:
        return [
            gamma - args.epsilon if args.epsilon == 0 else torch.norm(gamma) - args.epsilon
            for gamma in gammas
        ]
    return [torch.norm(gamma) - u for gamma, u in zip(gammas, us)]


def train(gpu, model, args, global_epoch):
    device = torch.device('cpu') if args.gpus == 0 else torch.device(f'cuda:{gpu}')
    is_cuda = device.type == 'cuda'

    model = model.to(device)

    dataset_train = NBodyDataset(partition='train', dataset_name=args.nbody_name,
                                 max_samples=args.max_samples)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)

    dataset_val = NBodyDataset(partition='val', dataset_name=args.nbody_name)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

    dataset_test = NBodyDataset(partition='test', dataset_name=args.nbody_name)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

    loss_mse = nn.MSELoss()
    transform = O3Transform(args.lmax_attr)

    optimizer_dual = None
    gammas = model.get_gammas()
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

    if args.optimizer == 'adam':
        optimizer = optim.Adam(primal_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(primal_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamsgd' and args.use_constrained:
        if args.resilience == 1:
            raise Exception("Resilience not supported with AdamSGD")

        model_params = [p for p in model.parameters() if all(p is not t for t in gammas)]

        optimizer = AdamSGD(model_params, gammas, args.lr, args.weight_decay)
    else:
        raise Exception("Optimizer not found")

    if args.use_constrained:

        lambdas = [
            torch.zeros(
                1, dtype=torch.float, requires_grad=False, device=device
            ).squeeze()
            for _ in range(no_gammas)
        ]

        if args.optimizer == 'adam':
            optimizer_dual = optim.Adam(lambdas, lr=args.dual_lr, weight_decay=0)
        elif args.optimizer == 'sgd':
            optimizer_dual = optim.SGD(lambdas, lr=args.dual_lr, weight_decay=0)
        else:
            optimizer_dual = optim.Adam(lambdas, lr=args.dual_lr, weight_decay=0)

    if args.log:
        wandb.init(project=args.wandb_proj_name, name=f"{args.wandb_prefix}_{args.model}_{args.dataset}_{args.target}_{args.lr}_d{args.dual_lr}_opt{args.optimizer}_auglg{args.auglag_const}_resil{args.resilience}_{args.ID}", config=vars(args), entity=args.wandb_entity)

    def log_metrics(metrics, commit=False, step=None, main_only=True):
        if not (args.log and args.wandb_log):
            return
        if main_only and gpu != 0:
            return
        wandb.log(metrics, commit=commit, step=step)

    best_val_loss = 1e8
    best_test_loss = 1e8
    best_test_loss_nonequi = 1e8
    best_val_loss_nonequi = 1e8
    best_epoch = 0

    for epoch in range(0, args.epochs):

        if is_cuda:
            torch.cuda.synchronize()
        start = time.time()
        train_loss, _ = run_epoch(
            model,
            optimizer,
            optimizer_dual,
            loss_mse,
            epoch,
            loader_train,
            transform,
            device,
            args,
            disable_equivariance=args.disable_equivariance,
            gammas=gammas,
            lambdas=lambdas,
            us=us,
            global_epoch=global_epoch,
            log_metrics_fn=log_metrics,
        )
        if is_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print("Time taken for epoch %d: %.2f seconds" % (epoch, end - start))

        log_metrics({"Train MSE": train_loss}, commit=False, step=global_epoch + epoch)

        if epoch % args.test_interval == 0 or epoch == args.epochs-1:
            start = time.time()

            val_loss, _ = run_epoch(
                model,
                optimizer,
                optimizer_dual,
                loss_mse,
                epoch,
                loader_val,
                transform,
                device,
                args,
                backprop=False,
                disable_equivariance=False,
                global_epoch=global_epoch,
                log_metrics_fn=log_metrics,
            )

            end = time.time()
            print("Time taken for val %d: %.2f seconds" % (epoch, end - start))

            test_loss, _ = run_epoch(
                model,
                optimizer,
                optimizer_dual,
                loss_mse,
                epoch,
                loader_test,
                transform,
                device,
                args,
                backprop=False,
                disable_equivariance=False,
                global_epoch=global_epoch,
                log_metrics_fn=log_metrics,
            )

            if args.disable_equivariance:
                val_loss_nonequi, _ = run_epoch(
                    model,
                    optimizer,
                    optimizer_dual,
                    loss_mse,
                    epoch,
                    loader_val,
                    transform,
                    device,
                    args,
                    backprop=False,
                    disable_equivariance=True,
                    global_epoch=global_epoch,
                    log_metrics_fn=log_metrics,
                )
                test_loss_nonequi, _ = run_epoch(
                    model,
                    optimizer,
                    optimizer_dual,
                    loss_mse,
                    epoch,
                    loader_test,
                    transform,
                    device,
                    args,
                    backprop=False,
                    disable_equivariance=True,
                    global_epoch=global_epoch,
                    log_metrics_fn=log_metrics,
                )
            
            if args.use_constrained and args.wandb_log:
                for i, gamma in enumerate(gammas):
                    log_metrics({f"gammas/Gamma_{i}": gamma.item()}, commit=False, step=global_epoch + epoch, main_only=False)
                
                for i, lambda_ in enumerate(lambdas):
                    log_metrics({f"lambdas/Lambda_{i}": lambda_.item()}, commit=False, step=global_epoch + epoch, main_only=False)

            if args.resilience == 1 and args.wandb_log:
                with torch.no_grad():
                    for i, u in enumerate(us):
                        log_metrics({f"us/U_{i}": u.item()}, commit=False, step=global_epoch + epoch, main_only=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                log_metrics({"Best Val MSE": best_val_loss}, commit=False, step=global_epoch + epoch)
                log_metrics({"Best Test MSE": best_test_loss}, commit=False, step=global_epoch + epoch)

            if args.disable_equivariance and val_loss_nonequi < best_val_loss_nonequi:
                best_val_loss_nonequi = val_loss_nonequi
                best_test_loss_nonequi = test_loss_nonequi
                log_metrics({"Best Val MSE Nonequi": best_val_loss_nonequi}, commit=False, step=global_epoch + epoch)
                log_metrics({"Best Test MSE Nonequi": best_test_loss_nonequi}, commit=False, step=global_epoch + epoch)

            log_metrics({"Val MSE": val_loss}, commit=False, step=global_epoch + epoch)
            log_metrics({"Test MSE": test_loss}, commit=False, step=global_epoch + epoch)
            if args.disable_equivariance:
                log_metrics({"Val MSE Nonequi": val_loss_nonequi}, commit=False, step=global_epoch + epoch)
                log_metrics({"Test MSE Nonequi": test_loss_nonequi}, commit=True, step=global_epoch + epoch)

            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" %
                  (best_val_loss, best_test_loss, best_epoch))
            print("Val Loss Equi: %.5f \t Test Loss Equi: %.5f" % (val_loss, test_loss))
            
            if args.disable_equivariance:
                print("Val Loss Nonequi: %.5f \t Test Loss Nonequi: %.5f" % (val_loss_nonequi, test_loss_nonequi))

    log_metrics({"Test MSE": best_test_loss})
    if args.disable_equivariance:
        log_metrics({"Test MSE Nonequi": best_test_loss_nonequi})

    if args.disable_equivariance:
        return best_val_loss, best_val_loss_nonequi, best_test_loss, best_test_loss_nonequi, best_epoch
    else:
        return best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, optimizer_dual, criterion, epoch, loader, transform, device, args, backprop=True, disable_equivariance=False, gammas=None, lambdas=None, us=None, global_epoch=0, log_metrics_fn=None):
    log_metrics_fn = log_metrics_fn or (lambda *_, **__: None)
    is_cuda = device.type == 'cuda'
    if backprop:
        model.train()
    else:
        model.eval()

    model.set_equivariance(disable_equivariance)

    res = {'epoch': epoch, 'loss': 0, 'loss_total': 0, 'counter': 0}
    n_nodes = 5
    batch_size = args.batch_size

    edges = loader.dataset.get_edges(args.batch_size, n_nodes)
    edges = [edges[0], edges[1]]
    edge_index = torch.stack(edges).long()
    batch_index_template = torch.arange(0, batch_size).repeat_interleave(n_nodes).long()

    equi_errors = []
    smallest_equi_difference = float('inf')
    biggest_equi_difference = 0.0

    for data in loader:
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, _, charges, loc_end = data
        if args.train_augments:
            R_train = Rotate(R=random_rotations(batch_size, device=loc.device))
            loc = R_train.transform_points(loc.reshape(batch_size, n_nodes, 3)).reshape(-1, 3)
            vel = R_train.transform_points(vel.reshape(batch_size, n_nodes, 3)).reshape(-1, 3)
            loc_end = R_train.transform_points(loc_end.reshape(batch_size, n_nodes, 3)).reshape(-1, 3)

        optimizer.zero_grad()

        if args.time_exp:
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.time()

        if args.model in {'segnn', 'seconv'}:
            graph = prepare_graph(loc, vel, charges, loc_end, edge_index, batch_index_template, device, transform)
            loc_pred = graph.pos + model(graph)
        else:
            raise Exception("Unknown model")

        if args.time_exp:
            if is_cuda:
                torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            if epoch % 100 == 99:
                print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))
                log_metrics_fn({"Time": time_exp_dic['time'] / time_exp_dic['counter']}, commit=False, step=global_epoch + epoch, main_only=False)

        if args.use_constrained and backprop:
            slacks = compute_slacks(gammas, args, us)

        loss = criterion(loc_pred, loc_end)
        loss_initial = loss.item()

        if args.use_constrained and args.resilience and backprop:
            loss += args.rho * torch.norm(torch.stack(us)) ** 2

        if args.use_constrained and backprop:
            dual_loss = 0.0

            for dual_var, slack in zip(lambdas, slacks):
                dual_loss += dual_var * slack + args.auglag_const * slack ** 2

            if args.resilience and backprop:
                for u, slack in zip(us, slacks):
                    dual_loss -= u * slack

            loss += dual_loss

            if args.resilience == 0:
                for ii, slack in enumerate(slacks):
                    lambdas[ii].grad = None if slack is None else -slack
            else:
                for ii, (slack, u) in enumerate(zip(slacks, us)):
                    lambdas[ii].grad = None if slack is None else -(slack - u)

        if backprop:
            loss.backward()
            optimizer.step()
            if args.use_constrained:
                optimizer_dual.step()
                if args.epsilon > 0 or args.resilience != 0:
                    for lambda_ in lambdas:
                        lambda_.clamp_(min=0)

        if args.log_equivariance_metric:
            with torch.no_grad():
                model.eval()
                n_transforms = 3

                graph = prepare_graph(loc, vel, charges, loc_end, edge_index, batch_index_template, device, transform)
                loc_pred_orig = model(graph)

                for _ in range(n_transforms):
                    R = Rotate(R=random_rotations(batch_size, device=loc.device))

                    loc_batched = loc.reshape(batch_size, n_nodes, 3)
                    loc_rot = R.transform_points(loc_batched).reshape(-1, 3)

                    vel_batched = vel.reshape(batch_size, n_nodes, 3)
                    vel_rot = R.transform_points(vel_batched).reshape(-1, 3)

                    graph = prepare_graph(loc_rot, vel_rot, charges, loc_end, edge_index, batch_index_template, device, transform)
                    loc_pred_rot = model(graph)

                    loc_pred_orig_rot = R.transform_points(loc_pred_orig.reshape(batch_size, n_nodes, 3))
                    loc_pred_rot = loc_pred_rot.reshape(batch_size, n_nodes, 3)

                    diff_equi = loc_pred_rot - loc_pred_orig_rot

                    min_value = diff_equi.pow(2).sqrt().min().item()
                    max_value = diff_equi.pow(2).sqrt().max().item()

                    smallest_equi_difference = min(smallest_equi_difference, min_value)
                    biggest_equi_difference = max(biggest_equi_difference, max_value)

                    diff_equi = diff_equi.norm(dim=-1).mean()
                    equi_errors.append(diff_equi.item())

        res['loss'] += loss_initial * batch_size
        res['loss_total'] += loss.item() * batch_size
        res['counter'] += batch_size

    if args.log_equivariance_metric and equi_errors:
        log_metrics_fn({"equi/error": float(np.mean(equi_errors))}, commit=False, step=global_epoch + epoch, main_only=False)
        log_metrics_fn({"equi/min": smallest_equi_difference}, commit=False, step=global_epoch + epoch, main_only=False)
        log_metrics_fn({"equi/max": biggest_equi_difference}, commit=False, step=global_epoch + epoch, main_only=False)

    prefix = "" if backprop else "==> "
    print('%s epoch %d avg loss: %.5f' % (prefix + loader.dataset.partition, epoch, res['loss'] / res['counter']))
    print('%s epoch %d avg total loss: %.5f' % (prefix + loader.dataset.partition, epoch, res['loss_total'] / res['counter']))

    return res['loss'] / res['counter'], res['loss_total'] / res['counter']