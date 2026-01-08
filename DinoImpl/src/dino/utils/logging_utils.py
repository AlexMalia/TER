"""Logging utilities."""

import logging
import sys
from pathlib import Path
from datetime import datetime

def get_logger_level(level_str: str) -> int:
    """Convert string log level to logging module level."""
    level_str = level_str.lower()
    if level_str == "debug":
        return logging.DEBUG
    elif level_str == "info":
        return logging.INFO
    elif level_str == "warning":
        return logging.WARNING
    elif level_str == "error":
        return logging.ERROR
    elif level_str == "critical":
        return logging.CRITICAL
    else:
        raise ValueError(f"Unknown log level: {level_str}")

def setup_logging(log_dir: str = './logs', log_level: str = "info") -> logging.Logger:
    """
    Setup logging configuration.

    Creates both console and file handlers with proper formatting.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (logging.INFO, logging.DEBUG, etc.)

    Returns:
        Root logger instance
    """
    log_level = get_logger_level(log_level)
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'train_{timestamp}.log'

    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")

    return logger


def log_metrics(metrics: dict, epoch: int, prefix: str = ""):
    """
    Log metrics in a formatted way.

    Args:
        metrics: Dictionary of metric names and values
        epoch: Current epoch
        prefix: Optional prefix for log message (e.g., "Train", "Val")
    """
    logger = logging.getLogger(__name__)
    metric_strs = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                   for k, v in metrics.items()]
    logger.info(f"{prefix} Epoch {epoch} - {', '.join(metric_strs)}")
