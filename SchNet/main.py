import argparse
import numpy as np
import torch
from layers.schnet import SchNet
from utils.training import evaluate, train
from datasets.datasets import get_dataset
import os
import wandb
import time

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_once(args, run_index=0, start_step=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataset(args)

    model = SchNet(
        hidden_channels=args.hidden_channels,
        num_filters=args.num_filters,
        num_interactions=args.num_interactions,
        cutoff=args.cutoff,
        readout="add",
        args=args,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_dual = None

    gammas = None
    lambdas = None

    if args.disable_equi:
        gammas = model.get_gammas()
        lambdas = [
            torch.zeros(1, dtype=torch.float, requires_grad=False, device=device).squeeze() for _ in gammas
        ]

        if args.use_constrained:
            optimizer_dual = torch.optim.Adam(lambdas, lr=args.lr_dual, weight_decay=0)

    # Create a run name (optional; used if saving checkpoints locally)
    run_name = (
        f"{args.prefix}_{args.dataset}_lrs({args.lr},{args.lr_dual})"
        f"_disable_equi_{args.disable_equi}_constrained_{args.use_constrained}"
        f"_run{run_index}"
    )

    best_val_metric_equi = float("inf")
    time1 = wandb.run.start_time if wandb.run else "offline_run"
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    best_checkpoint_path = os.path.join("checkpoints", f"{time1}_{run_name}.pth")

    for epoch in range(args.epochs):
        current_step = start_step + epoch  # global step for logging

        # --- Training ---
        torch.cuda.synchronize()
        start = time.time()
        loss_train, _, gammas_train = train(
            model,
            train_loader,
            optimizer,
            optimizer_dual,
            lambdas,
            gammas,
            device,
            args=args,
            constrained=args.use_constrained,
            disable_equivariance=args.disable_equi,
        )
        torch.cuda.synchronize()
        end = time.time()
        print(f"[Run {run_index}] Epoch: {epoch} | Train Time: {end - start} s")

        # --- Evaluate non-equiv if needed ---
        if args.disable_equi:
            metric_val_nonequi, std_val_nonequi, metric_name_nonequi, gammas_val_nonequi = evaluate(
                model, val_loader, device, args=args, disable_equivariance=True
            )
            print(
                f"[Run {run_index}] Epoch: {epoch:02d} | Train Loss: {loss_train:.6f} | "
                f"Val {metric_name_nonequi}(non-equiv): {metric_val_nonequi:.6f} ± {std_val_nonequi:.6f}"
            )
            wandb.log(
                {
                    "train_loss": loss_train,
                    f"val_{metric_name_nonequi}_non_equi": metric_val_nonequi,
                },
                commit=False,
                step=current_step,
            )
        else:
            print(f"[Run {run_index}] Epoch: {epoch:02d} | Train Loss: {loss_train:.6f}")

        # --- Evaluate equivariant ---
        torch.cuda.synchronize()
        start = time.time()
        metric_val_equi, std_val_equi, metric_name_equi, gammas_val_equi = evaluate(
            model, val_loader, device, args=args, disable_equivariance=False
        )
        torch.cuda.synchronize()
        end = time.time()

        print(f"[Run {run_index}] Epoch: {epoch} | Eval Time: {end - start} s")

        wandb.log(
            {f"val_{metric_name_equi}_equivariant": metric_val_equi},
            commit=False,
            step=current_step,
        )

        # --- Track best model (equiv) ---
        if metric_val_equi < best_val_metric_equi:
            best_val_metric_equi = metric_val_equi
            torch.save(model.state_dict(), best_checkpoint_path)

        if args.disable_equi:
            if args.n_gammas == 'nn':
                if gammas_train is not None:
                    for i, gamma in enumerate(gammas_train):
                        wandb.log({f"gammas_train/gamma_{i}": gamma.item()}, commit=False, step=current_step)
                if gammas_val_nonequi is not None:
                    for i, gamma in enumerate(gammas_val_nonequi):
                        wandb.log({f"gammas_val_nonequi/gamma_{i}": gamma.item()}, commit=False, step=current_step)
                if gammas_val_equi is not None:
                    for i, gamma in enumerate(gammas_val_equi):
                        wandb.log({f"gammas_val_equi/gamma_{i}": gamma.item()}, commit=False, step=current_step)
            else:
                for i, gamma in enumerate(gammas):
                    wandb.log({f"gammas/gamma_{i}": gamma.item()}, commit=False, step=current_step)

            for i, lambda_ in enumerate(lambdas):
                wandb.log({f"lambdas/lambda_{i}": lambda_.item()}, commit=False, step=current_step)

        # Ensure final commit for each epoch
        wandb.log({}, commit=True, step=current_step)

    # -----------------------------
    # Final test (load best checkpoint)
    # -----------------------------
    print(f"[Run {run_index}] Loading best model for final evaluation: {best_checkpoint_path}")
    best_state_dict = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_state_dict)

    # If disable_equi=True, test both modes
    if args.disable_equi:
        metric_test_nonequi, std_test_nonequi, metric_name_nonequi, _ = evaluate(
            model, test_loader, device, args=args, disable_equivariance=True
        )
        print(
            f"[Run {run_index}] Final Test {metric_name_nonequi} (non-equiv): "
            f"{metric_test_nonequi:.6f} ± {std_test_nonequi:.6f}"
        )
        wandb.log(
            {f"test_{metric_name_nonequi}_non_equi": metric_test_nonequi},
            step=start_step + args.epochs
        )

    # Always do the equivariant final test
    metric_test_equi, std_test_equi, metric_name_equi, _ = evaluate(
        model, test_loader, device, args=args, disable_equivariance=False
    )
    print(
        f"[Run {run_index}] Final Test {metric_name_equi} (equivariant): "
        f"{metric_test_equi:.6f} ± {std_test_equi:.6f}"
    )
    wandb.log(
        {f"test_{metric_name_equi}_equivariant": metric_test_equi},
        step=start_step + args.epochs
    )

    # Return final test metric + updated offset
    return metric_test_equi, (start_step + args.epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/QM9",
        help="Path to save/load the QM9 dataset",
    )
    parser.add_argument(
        "--target_idx",
        type=int,
        default=7,
        help="Which property index to train on [0..18]",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Cutoff distance for interatomic interactions",
    )
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--num_interactions", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_dual", type=float, default=1e-3)
    parser.add_argument("--use_constrained", type=int, default=1)
    parser.add_argument("--disable_equi", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="qm9", choices=["qm9"])
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--n_gammas", type=str, default="linear")
    parser.add_argument('--auglag_const', type=float, default=0., help='Augmented Lagrangian constant')
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of independent runs to execute",
    )
    args = parser.parse_args()

    if args.use_constrained and not args.disable_equi:
        raise ValueError("Constrained training requires disable_equi=1")

    # Initialize W&B once here; all runs share this logging
    run_name = (
        f"TASK{args.target_idx}_aug_"
        f"{args.prefix}_{args.dataset}_lrs({args.lr},{args.lr_dual})"
        f"_disable_equi_{args.disable_equi}_constrained_{args.use_constrained}_auglag_{args.auglag_const}"
        f"_numruns_{args.num_runs}"
    )
    wandb.init(
        project="constrained-qm9-redux",
        name=run_name,
        mode="online" if args.use_wandb else "disabled",
        entity="mls-2",
    )
    wandb.config.update(args)

    final_test_metrics = []
    current_step_offset = 0 

    for run_i in range(args.num_runs):
        seed = 42 + run_i
        set_seed(seed)

        print(f"========== Starting run {run_i+1}/{args.num_runs} with seed={seed} ==========")

        test_metric_equi, new_offset = run_once(
            args,
            run_index=run_i,
            start_step=current_step_offset
        )
        final_test_metrics.append(test_metric_equi)

        # Update our global step offset
        current_step_offset = new_offset

    final_test_metrics = np.array(final_test_metrics)
    mean_metric = final_test_metrics.mean()
    std_metric = final_test_metrics.std()

    print("========================================")
    print(f"Final test metrics (equivariant) over {args.num_runs} runs: {final_test_metrics}")
    print(f"Mean: {mean_metric:.6f}, Std: {std_metric:.6f}")
    print("========================================")

    wandb.log(
        {
            "final_test_metric_equi_mean": mean_metric,
            "final_test_metric_equi_std": std_metric,
        },
        step=current_step_offset
    )

    print("Done.")

if __name__ == "__main__":
    main()
