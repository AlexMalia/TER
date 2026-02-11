#!/usr/bin/env python3
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
    parser.add_argument("--dataset", type=str, help=f"Dataset name")
    parser.add_argument("--data-path", type=str, help=f"Path to data")
    parser.add_argument("--batch-size", type=int, help=f"Batch size")
    parser.add_argument("--num-workers", type=int, help=f"Number of data loading workers")

    # Override model config
    parser.add_argument("--backbone", type=str, help=f"Backbone architecture")
    parser.add_argument("--output-dim", type=int, help=f"Projection output dimension")

    # Override training config
    parser.add_argument("--epochs", type=int, help=f"Number of epochs")
    parser.add_argument("--lr", type=float, help=f"Learning rate")

    # Logging and checkpointing
    parser.add_argument("--checkpoint-dir", type=str, help=f"Checkpoint directory")
    parser.add_argument("--log-dir", type=str, help=f"Log directory")
    parser.add_argument("--log-verbosity", type=str, help=f"Logging verbosity level")

    # Other
    parser.add_argument("--seed", type=int, help=f"Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda if available, else cpu)")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    config = DinoConfig.from_yaml_and_args(args.config, args)  # Load config from YAML and override with command-line arguments

    # Setup logging
    setup_logging(config.logging_config.log_dir, config.logging_config.log_verbosity)
    logger = logging.getLogger(__name__)
    torch.set_printoptions(profile="full", linewidth=200)
    logger.info("Starting DINO training script")
    logger.info(f"Current device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Determine device
    device = args.device if torch.cuda.is_available() else "cpu"
    config.training_config.device = device

    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:\n{config}")

    # Set random seed
    torch.manual_seed(config.training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training_config.seed)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(config.data_config, config.augmentation_config)
    logger.info(f"Created dataloaders: {len(train_loader)} train batches")

    # Create models
    logger.info("Creating models...")
    student = DinoModel.from_config(config.model_config)
    teacher = DinoModel.from_config(config.model_config)

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
        config.loss_config,
        config.augmentation_config,
        out_dim=student.output_dim
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(student.parameters(), config.optimizer_config)

    accumulation_steps = config.training_config.gradient_accumulation_steps

    # Nombre de vraies updates par epoch (pas le nombre de forward passes)
    updates_per_epoch = len(train_loader) // accumulation_steps

    total_steps = config.training_config.num_epochs * updates_per_epoch
    warmup_steps = config.scheduler_config.warmup_epochs * updates_per_epoch

    scheduler = create_scheduler(
        optimizer,
        config.scheduler_config,
        config.optimizer_config,
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
    if config.checkpoint_config.resume_from:
        logger.info(f"Resuming from checkpoint: {config.checkpoint_config.resume_from}")
        trainer.resume_from_checkpoint(config.checkpoint_config.resume_from)
    elif args.resume:
        # Try to find latest checkpoint
        latest_checkpoint = find_latest_checkpoint(config.checkpoint_config.checkpoint_dir)
        if latest_checkpoint:
            logger.info(f"Found checkpoint: {latest_checkpoint}")
            trainer.resume_from_checkpoint(str(latest_checkpoint))

    # --- wandb setup ---
    if config.logging_config.use_wandb:
        try:
            import wandb

            # If resuming, try to get the wandb run ID from checkpoint
            wandb_run_id = getattr(trainer, 'resumed_wandb_run_id', None)
            if wandb_run_id is None:
                wandb_run_id = wandb.util.generate_id()

            wandb.init(
                project=config.logging_config.wandb_project or "dino-training",
                entity=config.logging_config.wandb_entity,
                name=config.logging_config.wandb_run_name,
                id=wandb_run_id,
                resume="allow",
                config=config.to_dict(),
                save_code=True,
                tags=[config.model_config.backbone, config.data_config.dataset],
            )

            # Define metric axes so iteration and epoch charts are separate
            wandb.define_metric("iteration")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="iteration")
            wandb.define_metric("epoch/*", step_metric="epoch")

            # Log computed values not in the static config
            wandb.config.update({
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "param_count": param_count['total'],
            }, allow_val_change=True)

        except ImportError:
            logger.warning("wandb not installed. pip install wandb to enable.")
            config.logging_config.use_wandb = False

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
    finally:
        if config.logging_config.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
