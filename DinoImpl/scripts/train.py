#!/usr/bin/env python3
"""
DINO Training Script

Train a DINO model for self-supervised learning.

Example usage:
    # Train with default config
    python scripts/train.py

    # Train with custom config
    python scripts/train.py --config configs/cifar100.yaml

    # Override specific parameters
    python scripts/train.py --config configs/imagenette.yaml --batch-size 64 --epochs 200

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/checkpoint_latest.pth
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dino.config.config import DinoConfig
from dino.data.dataloaders import create_dataloaders
from dino.models import DinoModel
from dino.loss import DinoLoss
from dino.training import DinoTrainer, create_optimizer, create_scheduler
from dino.utils import setup_logging, find_latest_checkpoint
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DINO model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    # Override data config
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--data-path", type=str, help="Path to data")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-workers", type=int, help="Number of data loading workers")

    # Override model config
    parser.add_argument("--backbone", type=str, help="Backbone architecture")
    parser.add_argument("--output-dim", type=int, help="Projection output dimension")

    # Override training config
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")

    # Logging and checkpointing
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, help="Log directory")
    parser.add_argument("--log-verbosity", type=str, help="Logging verbosity level")

    # Other
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = DinoConfig.from_yaml(args.config)

    # Override config with command-line arguments
    if args.dataset:
        config.data.dataset = args.dataset
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.num_workers:
        config.data.num_workers = args.num_workers
    if args.data_path:
        config.data.data_path = args.data_path
    if args.backbone:
        config.model.backbone = args.backbone
    if args.output_dim:
        config.model.projection_output_dim = args.output_dim
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.optimizer.lr = args.lr
    if args.checkpoint_dir:
        config.checkpoint.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.logging.log_dir = args.log_dir
    if args.log_verbosity:
        config.logging.log_verbosity = args.log_verbosity
    if args.seed:
        config.training.seed = args.seed
        config.data.seed = args.seed
    if args.resume:
        config.checkpoint.resume_from = args.resume

    # Setup logging
    setup_logging(config.logging.log_dir, config.logging.log_verbosity)
    logger = logging.getLogger(__name__)
    torch.set_printoptions(profile="full", linewidth=200)
    logger.info("Starting DINO training script")
    logger.info(f"Current device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Determine device
    device = args.device if torch.cuda.is_available() else "cpu"
    config.training.device = device

    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:\n{config}")

    # Set random seed
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(config)
    logger.info(f"Created dataloaders: {len(train_loader)} train batches")

    # Create models
    logger.info("Creating models...")
    student = DinoModel.from_config(config)
    teacher = DinoModel.from_config(config)

    # Teacher is a perfect copy of student at initialization
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    param_count = student.parameter_count()
    logger.info(
        f"Created models with {param_count['total']:,} parameters "
        f"({param_count['backbone']:,} backbone, {param_count['projection']:,} projection)"
    )

    # Create loss function
    dino_loss = DinoLoss.from_config(
        config.loss,
        config.augmentation,
        out_dim=student.output_dim
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(student.parameters(), config.optimizer)

    accumulation_steps = config.training.gradient_accumulation_steps

    # Nombre de vraies updates par epoch (pas le nombre de forward passes)
    updates_per_epoch = len(train_loader) // accumulation_steps

    total_steps = config.training.num_epochs * updates_per_epoch
    warmup_steps = config.scheduler.warmup_epochs * updates_per_epoch

    scheduler = create_scheduler(
        optimizer,
        config.scheduler,
        config.optimizer,
        total_steps,
        warmup_steps
    )

    logger.info(
        f"Created scheduler with {warmup_steps} warmup steps, {total_steps} total steps "
        f"(accumulation_steps={accumulation_steps})"
    )

    # Create trainer
    trainer = DinoTrainer(
        config=config,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=dino_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Resume from checkpoint if specified
    if config.checkpoint.resume_from:
        logger.info(f"Resuming from checkpoint: {config.checkpoint.resume_from}")
        trainer.resume_from_checkpoint(config.checkpoint.resume_from)
    elif args.resume:
        # Try to find latest checkpoint
        latest_checkpoint = find_latest_checkpoint(config.checkpoint.checkpoint_dir)
        if latest_checkpoint:
            logger.info(f"Found checkpoint: {latest_checkpoint}")
            trainer.resume_from_checkpoint(str(latest_checkpoint))

    # Train
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
