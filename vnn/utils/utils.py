from torch.optim import Optimizer
import torch
import logging
import math
import torch.optim as optim
import datetime

import shutil

from pathlib import Path


def setup_dir_and_logging(args):
    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    experiment_dir = Path("./log/")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath("cls")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath("logs/")
    log_dir.mkdir(parents=True, exist_ok=True)

    """LOG"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, experiment_dir, checkpoints_dir


def get_optimizer(args, parameters, lr, momentum=0):
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
    )
    else:
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
        )
    return optimizer

def get_scheduler(scheduler_type, optimizer, total_steps=None, num_warmup_steps=None, step_size=25, gamma=0.8, problem='primal', args=None):
    if scheduler_type == "cosine":
        if num_warmup_steps is None:
            num_warmup_steps = 0
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    elif scheduler_type == "cos_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.epoch,
            eta_min=args.lr if problem == "primal" else args.dual_lr,
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:
        logging.info('No scheduler selected.')
        scheduler = NoScheduler(optimizer)
    return scheduler

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int = 5,
    num_training_steps: int = 250,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class NoScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, *args, **kwargs):
        pass
