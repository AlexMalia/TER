"""DINO Trainer class."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
import logging
from tqdm import tqdm

from ..utils.ema import update_teacher_EMA, get_momentum_schedule
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.logging_utils import log_metrics

logger = logging.getLogger(__name__)


def get_global_local_views(view_set):
    """
    Separate global and local views from view set.

    Args:
        view_set: List of view tensors

    Returns:
        Tuple of (global_views, all_views)
    """
    global_views = view_set[:2]
    local_views = view_set
    return global_views, local_views


class DinoTrainer:
    """
    DINO Trainer for self-supervised learning.

    Handles the complete training loop including:
    - Forward passes through student and teacher
    - Loss computation
    - EMA updates for teacher
    - Checkpointing
    - Logging

    Args:
        config: Training configuration
        student: Student model
        teacher: Teacher model
        optimizer: Optimizer
        loss_fn: Loss function
        train_loader: Training data loader
        val_loader: Optional validation data loader
        device: Device to train on

    Example:
        >>> trainer = DinoTrainer(config, student, teacher, optimizer, loss_fn, train_loader)
        >>> trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        config,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        self.config = config
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Training state
        self.current_epoch = 0
        self.current_iteration = 0
        self.history = []

        # EMA momentum schedule
        if config.training.teacher_momentum_schedule:
            self.momentum_schedule = get_momentum_schedule(
                base_momentum=config.training.teacher_momentum,
                final_momentum=config.training.teacher_momentum_final,
                num_epochs=config.training.num_epochs,
                niter_per_epoch=len(train_loader)
            )
        else:
            self.momentum_schedule = None

        # Move models to device
        self.student.to(device)
        self.teacher.to(device)
        self.loss_fn.to(device)

        logger.info(f"Trainer initialized on device: {device}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of metrics for this epoch
        """
        self.student.train()
        self.teacher.eval()

        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.training.num_epochs}",
            leave=True
        )

        for batch_idx, (view_set, _) in enumerate(pbar):
            # Move views to device
            view_set = [v.to(self.device) for v in view_set]

            # Get global and all views
            global_views, all_views = get_global_local_views(view_set)

            # Forward pass - teacher only sees global views
            with torch.no_grad():
                teacher_outputs = [self.teacher(v) for v in global_views]
                teacher_output = torch.cat(teacher_outputs, dim=0)

            # Forward pass - student sees all views
            student_outputs = [self.student(v) for v in all_views]
            student_output = torch.cat(student_outputs, dim=0)

            # Compute loss
            loss = self.loss_fn(student_output, teacher_output)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    self.config.training.gradient_clip
                )

            # Optimizer step - train student
            self.optimizer.step()

            # EMA update for teacher
            if self.momentum_schedule is not None:
                momentum = self.momentum_schedule[self.current_iteration]
            else:
                momentum = self.config.training.teacher_momentum

            update_teacher_EMA(self.student, self.teacher, alpha=momentum)

            # Update statistics
            epoch_loss += loss.item()
            self.history.append(loss.item())
            self.current_iteration += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'momentum': f"{momentum:.4f}"})

            # Log periodically
            if (batch_idx + 1) % self.config.logging.log_every_n_iters == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                    f"Loss: {loss.item():.4f}, "
                    f"Momentum: {momentum:.4f}"
                )
                logger.debug(f"Loss infos: {self.loss_fn}")
                logger.debug(f"Updated center: {self.loss_fn.get_center()}")
                logger.debug(f"Gradient norms:")
                for name, param in self.student.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        logger.debug(f"  {name}: {grad_norm:.4f}")
        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        metrics = {
            'loss': avg_loss,
            'momentum': momentum if self.momentum_schedule else self.config.training.teacher_momentum
        }

        return metrics

    def train(self, num_epochs: Optional[int] = None):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train (uses config if not specified)
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Checkpoint directory: {self.config.checkpoint.checkpoint_dir}")

        for epoch in range(self.current_epoch + 1, self.current_epoch + num_epochs + 1):
            # Train for one epoch
            metrics = self.train_epoch(epoch)

            # Log metrics
            log_metrics(metrics, epoch, prefix="Train")

            # Save checkpoint
            if epoch % self.config.checkpoint.save_every_n_epochs == 0:
                save_checkpoint(
                    student=self.student,
                    teacher=self.teacher,
                    optimizer=self.optimizer,
                    dino_loss=self.loss_fn,
                    epoch=epoch,
                    iteration=self.current_iteration,
                    config=self.config,
                    metrics=metrics,
                    checkpoint_dir=self.config.checkpoint.checkpoint_dir
                )

            self.current_epoch = epoch

        logger.info("Training completed!")
        logger.info(f"Total iterations: {self.current_iteration}")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            student=self.student,
            teacher=self.teacher,
            optimizer=self.optimizer,
            dino_loss=self.loss_fn,
            device=self.device
        )

        self.current_epoch = checkpoint_info['epoch']
        self.current_iteration = checkpoint_info['iteration']

        logger.info(f"Resumed from epoch {self.current_epoch}, iteration {self.current_iteration}")
