"""DINO Trainer class."""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import LRScheduler
import os

from ..utils.ema import update_teacher_EMA, get_momentum_schedule
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.logging_utils import log_metrics
from ..utils.history import History
from ..evaluation import Evaluator

from ..config.config import DinoConfig
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
    """

    def __init__(
        self,
        config: DinoConfig,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LRScheduler],
        loss_fn: nn.Module,
        train_loader: DataLoader,
        device: str,
        train_eval_loader: Optional[DataLoader] = None,
        val_eval_loader: Optional[DataLoader] = None,
        evaluator: Optional[Evaluator] = None,
    ):
        self.config = config
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.device = device
        self.evaluator = evaluator

        self.train_eval_loader = train_eval_loader
        self.val_eval_loader = val_eval_loader

        # Training state
        self.current_epoch = 0
        self.current_iteration = 0
        self.history = History(metadata={
            'config': config.to_dict() if hasattr(config, 'to_dict') else str(config),
            'device': device,
        })

        
        self.n_iter_per_epoch = len(train_loader)

        # EMA momentum schedule
        if config.training_config.teacher_momentum_schedule:
            # Use optimizer steps per epoch (not raw batches) so the schedule
            # is traversed at the same rate as current_iteration is incremented.
            accumulation_steps = config.training_config.gradient_accumulation_steps
            updates_per_epoch = math.ceil(self.n_iter_per_epoch / accumulation_steps)
            self.momentum_schedule = get_momentum_schedule(
                base_momentum=config.training_config.teacher_momentum,
                final_momentum=config.training_config.teacher_momentum_final,
                num_epochs=config.training_config.num_epochs,
                niter_per_epoch=updates_per_epoch
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

        accumulation_steps = self.config.training_config.gradient_accumulation_steps
        num_batches = len(self.train_loader)

        epoch_loss = 0.0
        last_momentum = self._get_momentum()


        # Progress bar 
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self._target_epoch}",
            leave=True,
            total=num_batches,
        )

        self.optimizer.zero_grad()

        for batch_idx, (view_set, _) in enumerate(pbar):
            # Move views to device
            view_set = [v.to(self.device) for v in view_set]

            # Get global and all views
            global_views, all_views = get_global_local_views(view_set)

            # Forward pass - teacher only sees global views
            with torch.no_grad():
                teacher_output = torch.cat([self.teacher(v) for v in global_views], dim=0)

            # Forward pass - student sees all views
            student_output = torch.cat([self.student(v) for v in all_views], dim=0)

            # Compute loss
            loss = self.loss_fn(student_output, teacher_output)
            true_loss = loss.item()
            epoch_loss += true_loss

            (loss / accumulation_steps).backward()

            is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches

            if is_accumulation_step or is_last_batch:
                last_momentum = self._optimizer_step(true_loss)
            
            pbar.set_postfix({
                "loss": f"{true_loss:.4f}",
                "momentum": f"{last_momentum:.4f}",
            })

            # Log periodically
            if (batch_idx + 1) % self.config.logging_config.log_every_n_iters == 0:
                self._log_batch(epoch, batch_idx, num_batches, true_loss, last_momentum)
           
        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        return {
            "loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "momentum": last_momentum,
        }

    def train(self, num_epochs: Optional[int] = None):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train (uses config if not specified)
        """
        if num_epochs is None:
            num_epochs = self.config.training_config.num_epochs

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Checkpoint directory: {self.config.checkpoint_config.checkpoint_dir}")
        

        target_epoch = num_epochs or self.config.training_config.num_epochs
        self._target_epoch = target_epoch
        for epoch in range(self.current_epoch + 1, target_epoch + 1):
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

            if self.evaluator is not None and self.config.evaluation_config.use_knn_eval:
                if epoch % self.config.evaluation_config.eval_every_n_epochs == 0:
                    knn_metrics = self.evaluator.evaluate(
                        model=self.teacher,
                        train_loader=self.train_eval_loader,
                        test_loader=self.val_eval_loader
                    )

                    # Log KNN metrics
                    log_metrics(knn_metrics, epoch, prefix="KNN Eval")
                    self.history.record_evaluation(epoch, knn_metrics)
                    knn_plot_path = os.path.join(self.config.evaluation_config.knn_plot_dir, f"knn_eval_epoch_{epoch}.png")
                    self.evaluator.plot(self.teacher, self.val_eval_loader, save_path=knn_plot_path)
                    # wandb logging
                    if wandb is not None and wandb.run is not None:
                        wandb.log({
                            "epoch": epoch,
                            **{f"knn_eval/{k}": v for k, v in knn_metrics.items()},
                            "knn_eval/plot": wandb.Image(knn_plot_path)
                        })

            # Save checkpoint
            if epoch % self.config.checkpoint_config.save_every_n_epochs == 0:
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
                    checkpoint_dir=self.config.checkpoint_config.checkpoint_dir,
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



    def _get_momentum(self) -> float:
        """Return the momentum value for the current iteration."""
        if self.momentum_schedule is not None:
            idx = min(self.current_iteration, len(self.momentum_schedule) - 1)
            return self.momentum_schedule[idx]
        return self.config.training_config.teacher_momentum

    def _optimizer_step(self, true_loss: float) -> float:
        """
        Perform a full optimizer step:
          clip -> step -> zero_grad -> scheduler -> EMA update -> log metrics.

        Returns:
            The momentum value used for the EMA update.
        """
        if self.config.training_config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config.training_config.gradient_clip,
            )

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        momentum = self._get_momentum()
        update_teacher_EMA(self.student, self.teacher, alpha=momentum)

        current_lr = self.optimizer.param_groups[0]["lr"]
        self.history.record_iteration(
            iteration=self.current_iteration,
            metrics={"loss": true_loss, "learning_rate": current_lr, "momentum": momentum},
        )

        if wandb is not None and wandb.run is not None:
            wandb.log({
                "iteration": self.current_iteration,
                "train/loss": true_loss,
                "train/lr": current_lr,
                "train/momentum": momentum,
            })

        self.current_iteration += 1
        return momentum

    def _log_batch(
        self,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        true_loss: float,
        momentum: float,
    ) -> None:
        """Log per-batch diagnostics: training progress, teacher center health, and gradient flow."""

        # --- INFO: training progress for current batch ---
        logger.info(
            f"Epoch {epoch} [{batch_idx + 1}/{num_batches}] "
            f"Loss: {true_loss:.4f} | EMA Momentum (teacher update rate): {momentum:.4f}"
        )

        # --- DEBUG: internal state of DINO loss (sharpening temperatures, center norm) ---
        logger.debug(f"DINO loss internal state: {self.loss_fn}")

        # --- DEBUG: teacher output center vector (EMA-updated, prevents representation collapse) ---
        # The center is subtracted from teacher outputs before softmax to avoid trivial solutions
        # where the teacher outputs a constant distribution for all inputs.
        c = self.loss_fn.get_center()
        preview = ", ".join(f"{x:.3f}" for x in c.flatten()[:8].tolist())
        logger.debug(
            f"Teacher center (EMA centering, collapse prevention) | "
            f"shape={list(c.shape)} norm={c.norm():.4f} "
            f"μ={c.mean():.4f} σ={c.std():.4f} "
            f"range=[{c.min():.4f}, {c.max():.4f}] "
            f"preview=[{preview}, ...]"
        )

        # --- DEBUG: student gradient norms per block (detect vanishing/exploding gradients) ---
        grad_norms = {
            name: param.grad.data.norm(2).item()
            for name, param in self.student.named_parameters()
            if param.requires_grad and param.grad is not None
        }

        if grad_norms:
            total = sum(grad_norms.values())
            max_name = max(grad_norms, key=grad_norms.get)
            min_name = min(grad_norms, key=grad_norms.get)

            logger.debug(
                f"Student gradient norms (global) | "
                f"total={total:.4f} "
                f"max={grad_norms[max_name]:.4f} ({max_name}) "
                f"min={grad_norms[min_name]:.4f} ({min_name})"
            )

            # Aggregate by transformer block to assess gradient flow across depth
            # (shallow blocks like patch_embed should receive smaller gradients than head)
            block_norms: dict[str, list[float]] = {}
            for name, norm in grad_norms.items():
                prefix = name.split(".")[0]
                if prefix == "blocks":
                    prefix = ".".join(name.split(".")[:2])  # e.g. blocks.0, blocks.11
                block_norms.setdefault(prefix, []).append(norm)

            block_summary = " | ".join(
                f"{block}={sum(norms) / len(norms):.4f}"
                for block, norms in block_norms.items()
            )
            logger.debug(f"Student gradient norms (avg per block) | {block_summary}")