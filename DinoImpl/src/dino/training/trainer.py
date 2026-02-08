"""DINO Trainer class."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Union
import logging
from tqdm import tqdm

from ..utils.ema import update_teacher_EMA, get_momentum_schedule
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.logging_utils import log_metrics
from ..utils.history import History

try:
    import wandb
except ImportError:
    wandb = None

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
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        self.config = config
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Check if we're in streaming mode (IterableDataset doesn't have __len__)
        self.is_streaming = not hasattr(train_loader.dataset, '__len__')

        # Training state
        self.current_epoch = 0
        self.current_iteration = 0
        self.history = History(metadata={
            'config': config.to_dict() if hasattr(config, 'to_dict') else str(config),
            'device': device,
        })

        # Determine number of iterations per epoch
        if self.is_streaming:
            # For streaming, use config value or estimate based on ImageNet100 size
            streaming_samples = getattr(config.data, 'streaming_train_samples', None)
            if streaming_samples:
                self.niter_per_epoch = streaming_samples // config.data.batch_size
            else:
                # Default estimate for ImageNet100: ~117k train samples
                self.niter_per_epoch = 117000 // config.data.batch_size
            logger.info(f"Streaming mode: estimated {self.niter_per_epoch} iterations per epoch")
        else:
            self.niter_per_epoch = len(train_loader)

        # EMA momentum schedule
        if config.training.teacher_momentum_schedule:
            self.momentum_schedule = get_momentum_schedule(
                base_momentum=config.training.teacher_momentum,
                final_momentum=config.training.teacher_momentum_final,
                num_epochs=config.training.num_epochs,
                niter_per_epoch=self.niter_per_epoch
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
        num_batches = self.niter_per_epoch if self.is_streaming else len(self.train_loader)

        accumulation_steps = self.config.training.gradient_accumulation_steps

        # Progress bar (with total if known, otherwise streaming mode)
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.training.num_epochs}",
            leave=True,
            total=num_batches if not self.is_streaming else None
        )

        self.optimizer.zero_grad()

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

            loss = loss / accumulation_steps

            # Backward pass
            loss.backward()

            epoch_loss += loss.item()

            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.config.training.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.config.training.gradient_clip
                    )

                # Optimizer step - train student
                self.optimizer.step()

                # Reset gradients après la mise à jour
                self.optimizer.zero_grad()

                # Learning rate scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # EMA update for teacher
                if self.momentum_schedule is not None:
                    momentum = self.momentum_schedule[self.current_iteration]
                else:
                    momentum = self.config.training.teacher_momentum

                update_teacher_EMA(self.student, self.teacher, alpha=momentum)

                # Record iteration metrics (seulement lors des vraies updates)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history.record_iteration(
                    iteration=self.current_iteration,
                    metrics={
                        'loss': loss.item(),
                        'learning_rate': current_lr,
                        'momentum': momentum
                    }
                )
                self.current_iteration += 1

                # wandb iteration-level logging
                if wandb is not None and wandb.run is not None:
                    wandb.log({
                        "iteration": self.current_iteration,
                        "train/loss": loss.item(),
                        "train/lr": current_lr,
                        "train/momentum": momentum,
                    })

            # Update progress bar
            if self.momentum_schedule is not None:
                momentum = self.momentum_schedule[min(self.current_iteration, len(self.momentum_schedule) - 1)]
            else:
                momentum = self.config.training.teacher_momentum
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'momentum': f"{momentum:.4f}"})


             # Log periodically
            if (batch_idx + 1) % self.config.logging.log_every_n_iters == 0:
                progress_str = f"{batch_idx+1}" if self.is_streaming else f"{batch_idx+1}/{num_batches}"
                logger.info(
                    f"Epoch {epoch} [{progress_str}] "
                    f"Loss: {loss.item():.4f}, "
                    f"Momentum: {momentum:.4f}"
                )
                logger.debug(f"Loss infos: {self.loss_fn}")
                
                c = self.loss_fn.get_center()

                # 1. Format the first 5 elements for a quick preview
                preview = ", ".join([f"{x:.4f}" for x in c.flatten()[:10].tolist()])

                # 2. Log everything in one structured message
                logger.debug(
                    f"Center {list(c.shape)} | "
                    f"Norm: {c.norm():.4f} | "
                    f"μ: {c.mean():.4f} ± {c.std():.4f} | "
                    f"Range: [{c.min():.4f}, {c.max():.4f}] | "
                    f"Data: [{preview}, ...]"
                )

                logger.debug(f"Gradient norms:")
                for name, param in self.student.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        logger.debug(f"  {name}: {grad_norm:.4f}")

        if (batch_idx + 1) % accumulation_steps != 0:
            if self.config.training.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    self.config.training.gradient_clip
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.momentum_schedule is not None:
                momentum = self.momentum_schedule[self.current_iteration]
            else:
                momentum = self.config.training.teacher_momentum

            update_teacher_EMA(self.student, self.teacher, alpha=momentum)
            self.current_iteration += 1

        # Compute epoch metrics (use actual batch count for streaming)
        actual_batches = batch_idx + 1
        avg_loss = epoch_loss / actual_batches
        metrics = {
            'loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
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

            # Record epoch metrics
            self.history.record_epoch(epoch, metrics)

            # Log metrics
            log_metrics(metrics, epoch, prefix="Train")

            # wandb epoch-level logging
            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "epoch/avg_loss": metrics['loss'],
                    "epoch/lr": metrics['learning_rate'],
                    "epoch/momentum": metrics['momentum'],
                })

            # Save checkpoint
            if epoch % self.config.checkpoint.save_every_n_epochs == 0:
                wandb_run_id = None
                if wandb is not None and wandb.run is not None:
                    wandb_run_id = wandb.run.id

                save_checkpoint(
                    student=self.student,
                    teacher=self.teacher,
                    optimizer=self.optimizer,
                    dino_loss=self.loss_fn,
                    epoch=epoch,
                    iteration=self.current_iteration,
                    config=self.config,
                    metrics=metrics,
                    checkpoint_dir=self.config.checkpoint.checkpoint_dir,
                    history=self.history,
                    wandb_run_id=wandb_run_id,
                    scheduler=self.scheduler
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
            device=self.device,
        )

        self.current_epoch = checkpoint_info['epoch']
        self.current_iteration = checkpoint_info['iteration']

        # Fast-forward scheduler to the correct step instead of loading state_dict.
        # This avoids SequentialLR.load_state_dict() bugs that corrupt the LR on resume.
        if self.scheduler is not None and self.current_iteration > 0:
            logger.info(f"Fast-forwarding scheduler to iteration {self.current_iteration}...")
            for _ in range(self.current_iteration):
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Scheduler fast-forwarded, current LR: {lr:.6f}")

        # Restore history if available
        if checkpoint_info.get('history') is not None:
            self.history = checkpoint_info['history']

        # Expose wandb_run_id for callers to use during wandb.init
        self.resumed_wandb_run_id = checkpoint_info.get('wandb_run_id')

        logger.info(f"Resumed from epoch {self.current_epoch}, iteration {self.current_iteration}")
