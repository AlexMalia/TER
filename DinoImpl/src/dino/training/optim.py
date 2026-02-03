"""Optimizer and scheduler factory functions for DINO training."""

from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from dino.config import OptimizerConfig, SchedulerConfig


def create_optimizer(
    params: Iterator[torch.nn.Parameter],
    optimizer_config: OptimizerConfig
) -> Optimizer:
    """
    Create an optimizer from configuration.

    Args:
        params: Model parameters to optimize
        optimizer_config: Optimizer configuration dataclass

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer type is not supported
    """
    optimizer_name = optimizer_config.optimizer.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            momentum=optimizer_config.betas[0]  # Use first beta as momentum
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_config.optimizer}. "
            f"Available: adamw, adam, sgd"
        )


def create_scheduler(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfig,
    optimizer_config: OptimizerConfig,
    total_steps: int,
    warmup_steps: int
) -> LRScheduler:
    """
    Create a learning rate scheduler from configuration.

    Args:
        optimizer: The optimizer to schedule
        scheduler_config: Scheduler configuration dataclass
        optimizer_config: Optimizer configuration (for warmup start factor)
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps

    Returns:
        Configured scheduler instance

    Raises:
        ValueError: If scheduler type is not supported
    """
    scheduler_name = scheduler_config.scheduler.lower()

    if scheduler_name == "cosine_warmup":
        # Calculate warmup start factor
        start_factor = scheduler_config.warmup_start_lr / optimizer_config.lr \
            if scheduler_config.warmup_start_lr > 0 else 1e-8 / optimizer_config.lr

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=scheduler_config.min_lr
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=scheduler_config.min_lr
        )
    elif scheduler_name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=scheduler_config.min_lr / optimizer_config.lr,
            total_iters=total_steps
        )
    elif scheduler_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_config.scheduler}. "
            f"Available: cosine_warmup, cosine, linear, constant"
        )
