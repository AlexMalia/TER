#!/usr/bin/env python3
"""
DINO Training Script for Kaggle

This script runs on Kaggle's GPU environment.
It sets up the code from the uploaded dataset and runs training.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION - Change this to switch datasets
# ============================================================================

CONFIG_FILE = "kaggle-imagenette.yaml"    # for ImageNette
# CONFIG_FILE = "kaggle-imagenet100.yaml"   # for ImageNet100

RESUME_TRAINING = True  # Set to False to start fresh (ignore checkpoint dataset)

# ============================================================================
# PATHS
# ============================================================================

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Dataset paths
CODE_DATASET = KAGGLE_INPUT / "dino-code"
CHECKPOINT_DATASET = "dino-checkpoints"  # Name of your saved output dataset on Kaggle

# ============================================================================
# SETUP
# ============================================================================

def setup_environment():
    """Setup the training environment on Kaggle."""
    print("=" * 60)
    print("Setting up DINO training environment on Kaggle")
    print("=" * 60)

    # Show available datasets
    print("\nAvailable input datasets:")
    for item in KAGGLE_INPUT.iterdir():
        print(f"  - {item.name}")

    # Copy source code to working directory
    print("\nCopying source code to working directory...")
    src_dest = KAGGLE_WORKING / "src"
    if src_dest.exists():
        shutil.rmtree(src_dest)
    shutil.copytree(CODE_DATASET / "src", src_dest)

    # Copy configs
    configs_dest = KAGGLE_WORKING / "configs"
    if configs_dest.exists():
        shutil.rmtree(configs_dest)
    shutil.copytree(CODE_DATASET / "configs", configs_dest)

    # Copy scripts
    scripts_dest = KAGGLE_WORKING / "scripts"
    if scripts_dest.exists():
        shutil.rmtree(scripts_dest)
    shutil.copytree(CODE_DATASET / "scripts", scripts_dest)

    # Add src to Python path
    sys.path.insert(0, str(KAGGLE_WORKING / "src"))

    print(f"Source code copied to: {KAGGLE_WORKING}")


def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")

    # Copy pyproject.toml if available
    pyproject = CODE_DATASET / "pyproject.toml"
    if pyproject.exists():
        shutil.copy(pyproject, KAGGLE_WORKING / "pyproject.toml")

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision", "numpy", "pyyaml", "tqdm",
        "tensorboard", "scikit-learn", "matplotlib", "pillow",
        "transformers>=4.46.3", "wandb>=0.23.1"
    ], check=True)

    print("Dependencies installed.")


def check_gpu():
    """Check GPU availability."""
    import torch
    print("\n" + "=" * 60)
    print("GPU Information")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available! Training will be slow.")

    return "cuda" if torch.cuda.is_available() else "cpu"


def find_input_checkpoint():
    """
    Find the latest checkpoint from a previous run's saved output.

    On Kaggle, /kaggle/working/ is wiped between runs. To resume training:
    1. After a run, push checkpoint with: ./kaggle/kaggle_manager.sh push-checkpoint
    2. The checkpoint dataset is already added as kernel input
    3. Checkpoints appear at /kaggle/input/dino-checkpoints/

    Returns:
        Path to checkpoint file, or None if not found
    """
    import logging
    logger = logging.getLogger(__name__)

    # First, check the named checkpoint dataset (root, then checkpoints/ subdirectory)
    dataset_root = KAGGLE_INPUT / CHECKPOINT_DATASET
    for subpath in ["checkpoint_latest.pth", "checkpoints/checkpoint_latest.pth"]:
        named_path = dataset_root / subpath
        if named_path.exists():
            logger.info(f"Found checkpoint in '{CHECKPOINT_DATASET}' dataset: {named_path}")
            return named_path

    # Fallback: scan all input datasets for checkpoint files
    if KAGGLE_INPUT.exists():
        for dataset_dir in KAGGLE_INPUT.iterdir():
            if dataset_dir.name == CODE_DATASET.name:
                continue
            for subpath in ["checkpoint_latest.pth", "checkpoints/checkpoint_latest.pth"]:
                candidate = dataset_dir / subpath
                if candidate.exists():
                    logger.info(f"Found checkpoint in '{dataset_dir.name}' dataset: {candidate}")
                    return candidate

    logger.info("No input checkpoint found, starting from scratch")
    return None


def run_training(device: str):
    """Run the DINO training."""
    print("\n" + "=" * 60)
    print("Starting DINO Training")
    print("=" * 60)

    # Import training modules (after setup)
    from dino.config.config import DinoConfig
    from dino.data.dataloaders import create_dataloaders
    from dino.models import DinoModel
    from dino.loss import DinoLoss
    from dino.training import DinoTrainer, create_optimizer, create_scheduler
    from dino.utils import setup_logging
    import torch
    import logging

    # Load config - YAML is the single source of truth
    config_path = KAGGLE_WORKING / "configs" / CONFIG_FILE
    config = DinoConfig.from_yaml_and_args(str(config_path), args=None)

    # Only override device (detected at runtime)
    config.training_config.device = device

    # Create output directories from config
    Path(config.checkpoint_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging_config.log_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(config.logging_config.log_dir, config.logging_config.log_verbosity)
    logger = logging.getLogger(__name__)

    logger.info(f"Using config: {CONFIG_FILE}")
    logger.info(f"Configuration:\n{config}")

    # Set random seed
    torch.manual_seed(config.training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training_config.seed)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(config.data_config, config.augmentation_config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}")

    # Create models using factory method
    logger.info("Creating models...")
    student = DinoModel.from_config(config.model_config)
    teacher = DinoModel.from_config(config.model_config)

    # Initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    param_count = student.parameter_count()
    logger.info(
        f"Created models with {param_count['total']:,} parameters "
        f"({param_count['backbone']:,} backbone, {param_count['projection']:,} projection)"
    )

    # Create loss using factory method
    dino_loss = DinoLoss.from_config(
        config.loss_config,
        config.augmentation_config,
        out_dim=student.output_dim
    )

    # Create optimizer using factory method
    optimizer = create_optimizer(student.parameters(), config.optimizer_config)

    # Compute scheduler steps accounting for gradient accumulation
    accumulation_steps = config.training_config.gradient_accumulation_steps
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
        config=config.training_config,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=dino_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Resume from checkpoint if specified, or auto-detect from input datasets
    resume_path = config.checkpoint_config.resume_from
    if not resume_path and RESUME_TRAINING:
        found = find_input_checkpoint()
        if found:
            resume_path = str(found)
    elif not RESUME_TRAINING:
        logger.info("RESUME_TRAINING=False, starting fresh")

    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        trainer.resume_from_checkpoint(resume_path)

    # wandb setup for Kaggle
    if config.logging_config.use_wandb:
        try:
            import wandb

            # Read API key from code dataset (bundled by kaggle_manager.sh)
            api_key = None
            key_file = Path("/kaggle/input/dino-code/.wandb_key")
            if key_file.exists():
                api_key = key_file.read_text().strip()
                logger.info("Found WANDB_API_KEY from code dataset")
            if not api_key:
                api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)
            else:
                logger.warning("No wandb API key found, running in offline mode")
                os.environ["WANDB_MODE"] = "offline"

            # If resuming, reuse the wandb run ID from the checkpoint
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
                tags=[config.model_config.backbone, config.data_config.dataset, "kaggle"],
            )
            wandb.define_metric("iteration")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="iteration")
            wandb.define_metric("epoch/*", step_metric="epoch")
        except ImportError:
            logger.warning("wandb not installed, skipping")
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

    logger.info("Training completed!")

    # List output files
    print("\n" + "=" * 60)
    print("Output files (available for download):")
    print("=" * 60)

    checkpoint_dir = Path(config.checkpoint_config.checkpoint_dir)
    log_dir = Path(config.logging_config.log_dir)

    for dir_path in [checkpoint_dir, log_dir]:
        if dir_path.exists():
            print(f"\n{dir_path.name}/")
            for f in sorted(dir_path.rglob("*")):
                if f.is_file():
                    size_mb = f.stat().st_size / 1e6
                    print(f"  {f.relative_to(dir_path)} ({size_mb:.1f} MB)")


def main():
    """Main entry point."""
    setup_environment()
    install_dependencies()
    device = check_gpu()
    run_training(device)


if __name__ == "__main__":
    main()
