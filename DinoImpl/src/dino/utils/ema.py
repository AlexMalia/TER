"""Exponential Moving Average utilities for teacher network update."""

import torch
import torch.nn as nn
from typing import Optional
import math


@torch.no_grad()
def update_teacher_EMA(
    student: nn.Module,
    teacher: nn.Module,
    alpha: float = 0.99
):
    """
    Update teacher network with Exponential Moving Average of student weights.

    θ_teacher = α * θ_teacher + (1 - α) * θ_student

    Args:
        student: Student model
        teacher: Teacher model (will be updated in-place)
        alpha: EMA momentum coefficient (typical: 0.996-0.999)

    Example:
        >>> student = DinoModel(backbone, projection_head)
        >>> teacher = DinoModel(backbone_copy, projection_head_copy)
        >>> update_teacher_EMA(student, teacher, alpha=0.99)
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)


def get_momentum_schedule(
    base_momentum: float,
    final_momentum: float,
    num_epochs: int,
    niter_per_epoch: int
) -> list:
    """
    Get momentum schedule that increases from base to final momentum.

    Used in DINO to gradually increase teacher momentum during training
    (e.g., from 0.996 to 1.0).

    Args:
        base_momentum: Starting momentum
        final_momentum: Final momentum
        num_epochs: Total number of epochs
        niter_per_epoch: Number of iterations per epoch

    Returns:
        List of momentum values for each iteration

    Example:
        >>> schedule = get_momentum_schedule(0.996, 1.0, num_epochs=100, niter_per_epoch=200)
        >>> print(len(schedule))
        20000
        >>> print(schedule[0], schedule[-1])
        0.996 1.0
    """
    momentum_schedule = []
    for epoch in range(num_epochs):
        for _ in range(niter_per_epoch):
            # Cosine schedule from base_momentum to final_momentum
            progress = epoch * niter_per_epoch + len(momentum_schedule)
            total_iters = num_epochs * niter_per_epoch
            momentum = final_momentum - (final_momentum - base_momentum) * (
                math.cos(math.pi * progress / total_iters) + 1
            ) / 2
            momentum_schedule.append(momentum)

    return momentum_schedule


class EMAUpdater:
    """
    Helper class for managing EMA updates with optional scheduling.

    Args:
        momentum: Base EMA momentum
        momentum_schedule: Optional list of momentum values per iteration
        start_iter: Starting iteration (for resuming training)

    Example:
        >>> updater = EMAUpdater(momentum=0.996)
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         # ... train student ...
        ...         updater.update(student, teacher)
    """

    def __init__(
        self,
        momentum: float = 0.996,
        momentum_schedule: Optional[list] = None,
        start_iter: int = 0
    ):
        self.base_momentum = momentum
        self.momentum_schedule = momentum_schedule
        self.iter = start_iter

    def update(self, student: nn.Module, teacher: nn.Module):
        """Update teacher with EMA."""
        if self.momentum_schedule is not None:
            # Use scheduled momentum
            if self.iter < len(self.momentum_schedule):
                momentum = self.momentum_schedule[self.iter]
            else:
                momentum = self.momentum_schedule[-1]
        else:
            momentum = self.base_momentum

        update_teacher_EMA(student, teacher, alpha=momentum)
        self.iter += 1

    def get_current_momentum(self) -> float:
        """Get current momentum value."""
        if self.momentum_schedule is not None:
            if self.iter < len(self.momentum_schedule):
                return self.momentum_schedule[self.iter]
            return self.momentum_schedule[-1]
        return self.base_momentum

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            'base_momentum': self.base_momentum,
            'iter': self.iter,
            'momentum_schedule': self.momentum_schedule
        }

    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        self.base_momentum = state_dict['base_momentum']
        self.iter = state_dict['iter']
        self.momentum_schedule = state_dict.get('momentum_schedule')
