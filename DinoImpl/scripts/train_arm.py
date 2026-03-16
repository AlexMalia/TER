#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dino.config.config import DinoConfig
from dino.data.dataloaders import create_train_dataloaders, create_eval_dataloaders
from dino.models import DinoModel
from dino.loss import DinoLoss
from dino.training import DinoTrainer, create_optimizer, create_scheduler
from dino.utils import setup_logging, find_latest_checkpoint
from dino.evaluation.knn import KNNClassifier

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
    if config.training_config.seed:
        torch.manual_seed(config.training_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.training_config.seed)
        np.random.seed(config.training_config.seed)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_train_dataloaders(config.data_config, config.augmentation_config, is_graph=True)
    logger.info(f"Created dataloaders: {len(train_loader)} train batches")

    


if __name__ == "__main__":
    main()
