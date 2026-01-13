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
from dino.models.backbone import get_backbone
from dino.models import get_projection_head, DinoModel
from dino.loss import DinoLoss
from dino.training import DinoTrainer
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
    backbone_student = get_backbone(
        config.model.backbone,
        pretrained=config.model.backbone_pretrained
    )

    projection_head_student = get_projection_head(
        input_dim=backbone_student.output_dim,
        output_dim=config.model.projection_output_dim,
        hidden_dim=config.model.projection_hidden_dim,
        bottleneck_dim=config.model.projection_bottleneck_dim,
        use_weight_norm=config.model.use_weight_norm
    )

    student = DinoModel(backbone_student, projection_head_student)

    # Create teacher as copy of student
    backbone_teacher = get_backbone(
        config.model.backbone,
        pretrained=config.model.backbone_pretrained
    )

    projection_head_teacher = get_projection_head(
        input_dim=backbone_teacher.output_dim,
        output_dim=config.model.projection_output_dim,
        hidden_dim=config.model.projection_hidden_dim,
        bottleneck_dim=config.model.projection_bottleneck_dim,
        use_weight_norm=config.model.use_weight_norm
    )

    teacher = DinoModel(backbone_teacher, projection_head_teacher)

    # Initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    param_count = student.parameter_count()
    logger.info(
        f"Created models with {param_count['total']:,} parameters "
        f"({param_count['backbone']:,} backbone, {param_count['projection']:,} projection)"
    )

    # Create loss function
    n_global_crops = 2
    ncrops = n_global_crops + config.augmentation.num_local_views

    dino_loss = DinoLoss(
        out_dim=config.model.projection_output_dim,
        student_temp=config.loss.student_temp,
        teacher_temp=config.loss.teacher_temp,
        center_momentum=config.loss.center_momentum,
        n_global_crops=n_global_crops,
        ncrops=ncrops
    )

    # Create optimizer
    if config.optimizer.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.optimizer}")

    # Create trainer
    trainer = DinoTrainer(
        config=config,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
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
