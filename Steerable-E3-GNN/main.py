import torch
import argparse
import os
import numpy as np
import torch.multiprocessing as mp
from e3nn.o3 import Irreps, spherical_harmonics
from models.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
import wandb

def _find_free_port():
    """ Find free port, so multiple runs don't clash """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--print', type=int, default=100,
                        help='print interval')
    parser.add_argument('--log', type=bool, default=True,
                        help='logging flag')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers in dataloader')
    parser.add_argument('--save_dir', type=str, default="saved models",
                        help='Directory in which to save models')

    # Data parameters
    parser.add_argument('--dataset', type=str, default="qm9",
                        help='Data set')
    parser.add_argument('--root', type=str, default="datasets",
                        help='Data set location')
    parser.add_argument('--download', type=bool, default=False,
                        help='Download flag')

    # QM9 parameters
    parser.add_argument('--target', type=str, default="alpha",
                        help='Target value, also used for gravity dataset [pos, force]')
    parser.add_argument('--radius', type=float, default=2,
                        help='Radius (Angstrom) between which atoms to add links.')
    parser.add_argument('--feature_type', type=str, default="one_hot",
                        help='Type of input feature: one-hot, or Cormorants charge thingy')

    # Nbody parameters:
    parser.add_argument('--nbody_name', type=str, default="nbody_small",
                        help='Name of nbody data [nbody, nbody_small]')
    parser.add_argument('--max_samples', type=int, default=3000,
                        help='Maximum number of samples in nbody dataset')
    parser.add_argument('--time_exp', type=bool, default=False,
                        help='Flag for timing experiment')
    parser.add_argument('--test_interval', type=int, default=5,
                        help='Test every test_interval epochs')

    # Gravity parameters:
    parser.add_argument('--neighbours', type=int, default=6,
                        help='Number of connected nearest neighbours')

    # Model parameters
    parser.add_argument('--model', type=str, default="segnn",
                        help='Model name')
    parser.add_argument('--hidden_features', type=int, default=64,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_h', type=int, default=1,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_attr', type=int, default=1,
                        help='max degree of geometric attribute embedding')
    parser.add_argument('--subspace_type', type=str, default="weightbalanced",
                        help='How to divide spherical harmonic subspaces')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of message passing layers')
    parser.add_argument('--norm', type=str, default="instance",
                        help='Normalisation type [instance, batch]')
    parser.add_argument('--pool', type=str, default="avg",
                        help='Pooling type type [avg, sum]')
    parser.add_argument('--conv_type', type=str, default="linear",
                        help='Linear or non-linear aggregation of local information in SEConv')
    
    parser.add_argument('--disable_equivariance', type=int, default=1,
                        help='Disable equivariance')
    parser.add_argument('--use_constrained', type=int, default=1,
                        help='Constrained training')
    parser.add_argument('--epsilon', type=float, default=0.,
                        help='Constrained training epsilon')
    parser.add_argument('--dual_lr', type=float, default=5e-4,
                        help='Dual learning rate')
    
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer')
    parser.add_argument('--auglag_const', type=float, default=0., help='Augmented Lagrangian constant')

    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use (assumes all are on one node)')

    # New arguments:
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of runs for the experiment')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Global epoch offset for logging')
    
    parser.add_argument('--log_equivariance_metric', type=int, default=0,
                        help='Log equivariance metric')
    
    # Note that setting the seed does not guarantee perfect reproducibility; there are some other sources of randomness
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--resilience', type=int, default=0, help='If 1, use the resilience constraint')

    parser.add_argument('--rho', type=int, default=1, help='The weight of the resilience constraint')
    
    parser.add_argument('--train_augments', type=int, default=0, help='If 1, use SO3 augments during training')
    
    parser.add_argument('--wandb_prefix', type=str, default="",
                        help='Prefix for wandb run name')
    
    parser.add_argument('--wandb_log', type=bool, default=True,
                        help='Wandb logging flag')
    
    parser.add_argument('--wandb_proj_name', type=str, default="nbody-reprod",
                        help='Wandb project name')
    
    parser.add_argument('--wandb_entity', type=str, default="mls-2",
                        help='Wandb entity name')

    args = parser.parse_args()
    print(args)

    if args.dataset == 'qm9':
        raise NotImplementedError("QM9 is not currently verified; please use n-body and check for updates later.")

    # Initialize our global epoch counter
    global_epoch = args.start_epoch

    # Containers for aggregated results
    all_best_val_losses = []
    all_best_test_losses = []
    all_best_epochs = []
    all_best_val_losses_nonequi = []
    all_best_test_losses_nonequi = []

    # Loop over runs
    for run in range(args.n_runs):
        seed = args.seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Run {run+1}/{args.n_runs} with seed {seed}")
        
        # Set current global offset for this run
        args.start_epoch = global_epoch

        # --- Dataset, model, and training setup (as before) ---
        if args.dataset == "qm9":
            from qm9.train import train
            task = "graph"
            if args.feature_type == "one_hot":
                input_irreps = Irreps("5x0e")
            elif args.feature_type == "cormorant":
                input_irreps = Irreps("15x0e")
            elif args.feature_type == "gilmer":
                input_irreps = Irreps("11x0e")
            output_irreps = Irreps("1x0e")
            edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
            node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
            additional_message_irreps = Irreps("1x0e")
        elif args.dataset == "nbody":
            from nbody.train_nbody import train
            task = "node"
            input_irreps = Irreps("2x1o + 1x0e")
            output_irreps = Irreps("1x1o")
            edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
            node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
            additional_message_irreps = Irreps("2x0e")
        elif args.dataset == "gravity":
            from nbody.train_gravity import train
            task = "node"
            input_irreps = Irreps("2x1o + 1x0e")
            output_irreps = Irreps("1x1o")
            edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
            node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
            additional_message_irreps = Irreps("2x0e")
        else:
            raise Exception("Dataset could not be found")

        # Create hidden irreps
        if args.subspace_type == "weightbalanced":
            hidden_irreps = WeightBalancedIrreps(
                Irreps("{}x0e".format(args.hidden_features)), node_attr_irreps, sh=True, lmax=args.lmax_h)
        elif args.subspace_type == "balanced":
            hidden_irreps = BalancedIrreps(args.lmax_h, args.hidden_features, True)
        else:
            raise Exception("Subspace type not found")

        # Select model
        if args.model == "segnn":
            from models.segnn.segnn import SEGNN
            model = SEGNN(input_irreps,
                          hidden_irreps,
                          output_irreps,
                          edge_attr_irreps,
                          node_attr_irreps,
                          num_layers=args.layers,
                          norm=args.norm,
                          pool=args.pool,
                          task=task,
                          additional_message_irreps=additional_message_irreps,
                          args=args)
            args.ID = "_".join([args.model, args.dataset, args.target, str(np.random.randint(1e4, 1e5)), f"run{run}", f"constraint{args.use_constrained}", f"resilience{args.resilience}"])
        elif args.model == "seconv":
            from models.segnn.seconv import SEConv
            model = SEConv(input_irreps,
                           hidden_irreps,
                           output_irreps,
                           edge_attr_irreps,
                           node_attr_irreps,
                           num_layers=args.layers,
                           norm=args.norm,
                           pool=args.pool,
                           task=task,
                           additional_message_irreps=additional_message_irreps,
                           conv_type=args.conv_type)
            args.ID = "_".join([args.model, args.conv_type, args.dataset, str(np.random.randint(1e4, 1e5)), f"run{run}", f"constraint{args.use_constrained}", f"resilience{args.resilience}"])
        else:
            raise Exception("Model could not be found")

        print(model)
        print("The model has {:,} parameters.".format(sum(p.numel() for p in model.parameters())))
        if args.gpus == 0:
            print('Starting training on the cpu...')
            args.mode = 'cpu'
            if args.dataset == "nbody" and args.disable_equivariance == 1:
                best_val_loss, best_val_loss_nonequi, best_test_loss, best_test_loss_nonequi, best_epoch = train(0, model, args, global_epoch)
            else:
                best_val_loss, best_test_loss, best_epoch = train(0, model, args, global_epoch)
        elif args.gpus == 1:
            print('Starting training on a single gpu...')
            args.mode = 'gpu'
            if args.dataset == "nbody" and args.disable_equivariance == 1:
                best_val_loss, best_val_loss_nonequi, best_test_loss, best_test_loss_nonequi, best_epoch = train(0, model, args, global_epoch)
            else:
                best_val_loss, best_test_loss, best_epoch = train(0, model, args, global_epoch)
        elif args.gpus > 1:
            print('Starting training on', args.gpus, 'gpus...')
            args.mode = 'gpu'
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            port = _find_free_port()
            print('found free port', port)
            os.environ['MASTER_PORT'] = str(port)
            if args.dataset == "nbody" and args.disable_equivariance == 1:
                best_val_loss, best_val_loss_nonequi, best_test_loss, best_test_loss_nonequi, best_epoch = mp.spawn(train, nprocs=args.gpus, args=(model, args, global_epoch))
            else:
                best_val_loss, best_test_loss, best_epoch = mp.spawn(train, nprocs=args.gpus, args=(model, args,))


        all_best_val_losses.append(best_val_loss)
        all_best_test_losses.append(best_test_loss)
        all_best_epochs.append(best_epoch)

        if args.disable_equivariance and args.dataset == "nbody":
            all_best_test_losses_nonequi.append(best_test_loss_nonequi)
            all_best_val_losses_nonequi.append(best_val_loss_nonequi)
            
        global_epoch += args.epochs + 1

    # After all runs, compute and log aggregate results
    mean_val = np.mean(all_best_val_losses)
    std_val = np.std(all_best_val_losses)
    mean_test = np.mean(all_best_test_losses)
    std_test = np.std(all_best_test_losses)
    mean_epoch = np.mean(all_best_epochs)

    print("Aggregate Results:")
    print(f"Best Val Loss: Mean = {mean_val:.5f}, Std = {std_val:.5f}")
    print(f"Best Test Loss: Mean = {mean_test:.5f}, Std = {std_test:.5f}")
    print(f"Best Epoch: Mean = {mean_epoch:.1f}")

    if args.disable_equivariance == 1 and args.dataset == "nbody":
        mean_val_nonequi = np.mean(all_best_val_losses_nonequi)
        std_val_nonequi = np.std(all_best_val_losses_nonequi)
        mean_test_nonequi = np.mean(all_best_test_losses_nonequi)
        std_test_nonequi = np.std(all_best_test_losses_nonequi)

        print("Aggregate Results (Nonequi):")
        print(f"Best Val Loss: Mean = {mean_val_nonequi:.5f}, Std = {std_val_nonequi:.5f}")
        print(f"Best Test Loss: Mean = {mean_test_nonequi:.5f}, Std = {std_test_nonequi:.5f}")