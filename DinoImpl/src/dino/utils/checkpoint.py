"""Checkpoint management utilities."""

import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .history import History

logger = logging.getLogger(__name__)


def save_checkpoint(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dino_loss: torch.nn.Module,
    epoch: int,
    iteration: int,
    config: Any,
    metrics: Optional[Dict[str, float]] = None,
    checkpoint_dir: str = './checkpoints',
    filename: Optional[str] = None,
    is_best: bool = False,
    history: Optional['History'] = None,
    wandb_run_id: Optional[str] = None,
    scheduler: Optional[Any] = None
) -> Path:
    """
    Save a complete checkpoint of the DINO training state.

    Args:
        student: Student model
        teacher: Teacher model
        optimizer: Optimizer
        dino_loss: Loss module (contains center)
        epoch: Current epoch
        iteration: Current iteration
        config: Training configuration
        metrics: Optional metrics dict
        checkpoint_dir: Directory to save checkpoints
        filename: Optional custom filename
        is_best: Whether this is the best checkpoint
        history: Optional History instance to save alongside checkpoint

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        filename = f'checkpoint_epoch_{epoch:04d}.pth'

    checkpoint_path = checkpoint_dir / filename

    # Prepare checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'student_state_dict': student.state_dict(),
        'teacher_state_dict': teacher.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dino_loss_center': dino_loss.center,
        'config': config.to_dict() if hasattr(config, 'to_dict') else config,
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat(),
        'has_history': history is not None,
        'wandb_run_id': wandb_run_id,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Also save a "latest" checkpoint
    latest_path = checkpoint_dir / 'checkpoint_latest.pth'
    torch.save(checkpoint, latest_path)

    # Save history to separate JSON file
    if history is not None:
        history_filename = checkpoint_path.stem + '_history.json'
        history_path = checkpoint_dir / history_filename
        history.save(history_path)

        # Also save latest history
        latest_history_path = checkpoint_dir / 'history_latest.json'
        history.save(latest_history_path)

    # Save best checkpoint if specified
    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"Best checkpoint saved to {best_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dino_loss: torch.nn.Module,
    device: str = 'cpu',
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Load checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        student: Student model to load weights into
        teacher: Teacher model to load weights into
        optimizer: Optimizer to load state into
        dino_loss: Loss module to restore center
        device: Device to map checkpoint to

    Returns:
        Dictionary with checkpoint info (epoch, iteration, metrics, etc.)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model states
    student.load_state_dict(checkpoint['student_state_dict'])
    teacher.load_state_dict(checkpoint['teacher_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore loss center (ensure it's on the correct device)
    dino_loss.center = checkpoint['dino_loss_center'].to(device)

    # Restore scheduler state if available
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Scheduler state restored from checkpoint")

    # Get training info
    epoch = checkpoint['epoch']
    iteration = checkpoint.get('iteration', 0)
    metrics = checkpoint.get('metrics', {})

    logger.info(
        f"Checkpoint loaded successfully! "
        f"Epoch: {epoch}, Iteration: {iteration}"
    )

    # Try to load history from associated JSON file
    history = None
    if checkpoint.get('has_history', False):
        from .history import History
        history_filename = checkpoint_path.stem + '_history.json'
        history_path = checkpoint_path.parent / history_filename
        if history_path.exists():
            history = History.load(history_path)
        else:
            # Try loading latest history as fallback
            latest_history_path = checkpoint_path.parent / 'history_latest.json'
            if latest_history_path.exists():
                history = History.load(latest_history_path)

    return {
        'epoch': epoch,
        'iteration': iteration,
        'metrics': metrics,
        'config': checkpoint.get('config'),
        'timestamp': checkpoint.get('timestamp'),
        'history': history,
        'wandb_run_id': checkpoint.get('wandb_run_id'),
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """Find the latest checkpoint in directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # First try checkpoint_latest.pth
    latest_path = checkpoint_dir / 'checkpoint_latest.pth'
    if latest_path.exists():
        return latest_path

    # Otherwise find most recent checkpoint file
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def list_checkpoints(checkpoint_dir: str = './checkpoints') -> list:
    """
    List all available checkpoints in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of checkpoint file paths sorted by modification time
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return []

    checkpoints = sorted(
        checkpoint_dir.glob('checkpoint_*.pth'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    logger.info(f"Found {len(checkpoints)} checkpoint(s) in {checkpoint_dir}:")
    for cp in checkpoints:
        size_mb = cp.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(cp.stat().st_mtime)
        logger.info(
            f"  {cp.name:40s} - {size_mb:6.2f} MB - "
            f"{mtime.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    return checkpoints
