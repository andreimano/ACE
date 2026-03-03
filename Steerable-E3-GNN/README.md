# E(3) Steerable GNN

This folder contains the ACE variant of the model from
[Geometric and Physical Quantities improve E(3) Equivariant Message Passing](https://arxiv.org/abs/2110.02905) by Johannes Brandstetter, Rob Hesselink, Elise van der Pol, Erik Bekkers and Max Welling.

To reproduce the QM9 experiments (e.g. for seed 42 and target zpve), run:
`python3 main.py --dataset=qm9 --target=zpve --radius=2 --model=segnn --lmax_h=2 --lmax_attr=3 --layers=5 --subspace_type=weightbalanced --norm=instance --gpu=1 --weight_decay=0 --pool=avg --disable_equivariance 1 --use_constrained 1 --resilience 0 --num_workers 1 --batch_size=1024 --lr=6e-4 --dual_lr 1e-2 --epochs=600 --n_runs=1 --seed 42`

To reproduce the N-Body ACE experiments, run:

`python3 main.py --dataset=nbody --max_samples=3000 --norm=none --optimizer 'adam' --lr 9e-4 --dual_lr 8e-4 --auglag_const 0. --disable_equivariance 1 --use_constrained 1 --resilience 0 --log_equivariance_metric 0 --rho 1`

To run the baseline experiments either use the the official [SEGNN implementation](https://github.com/RobDHess/Steerable-E3-GNN) or set `disable_equivariance` and `use_constrained` to `0`. Note that running the N-Body baseline from here will obtain better results than the official implementation, for more details please see [this issue](https://github.com/RobDHess/Steerable-E3-GNN/issues/12).

To run the Figure 2 (Right) experiments, set the number of samples using `max_samples`.

Note that we use wandb to log experiments. For plots similar to Figure 2, fetch the data from wandb by using the API:

```
equality_run_id = "YOUR_WANDB_PROJECT/RUN_ID"      # "Equality Constraint"
no_constraints_run_id = "YOUR_WANDB_PROJECT/RUN2_ID"  # "No Constraints"

api = wandb.Api()
run_equal = api.run(equality_run_id)
run_no_const = api.run(no_constraints_run_id)

def extract_val_mse(run, key="Val MSE"):
    history = list(run.scan_history(keys=[key]))
    values = [row[key] for row in history if row.get(key) is not None]
    print(f"Run {run.id}: extracted {len(values)} points for {key}.")
    return values

val_mse_equal = extract_val_mse(run_equal, "Val MSE")
val_mse_no_const = extract_val_mse(run_no_const, "Val MSE")
```