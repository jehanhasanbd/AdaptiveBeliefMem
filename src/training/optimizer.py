# src/training/optimizer.py (fixed)
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from typing import Tuple


def create_optimizer(model_params, config: dict) -> Tuple[AdamW, LinearLR]:
    """Create optimizer and scheduler with phase-based learning rates."""

    # Ensure learning rate is float
    lr = config["training"]["learning_rate"]
    if isinstance(lr, str):
        lr = float(lr)

    weight_decay = config["training"]["weight_decay"]
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)

    warmup_steps = config["training"]["warmup_steps"]
    if isinstance(warmup_steps, str):
        warmup_steps = int(warmup_steps)

    total_steps = config["training"]["num_epochs"] * 1000  # Estimate
    if isinstance(total_steps, str):
        total_steps = int(total_steps)

    optimizer = AdamW(
        model_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    return optimizer, scheduler


def get_phase_lr(phase: str) -> float:
    """Get learning rate for training phase."""
    phase_lrs = {
        "warmup": 1e-4,
        "stabilization": 5e-5,
        "fine_tuning": 1e-5
    }
    return phase_lrs.get(phase, 2e-5)