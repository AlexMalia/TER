"""
DINO: Self-Supervised Learning with Knowledge Distillation

A PyTorch implementation of DINO (Self-DIstillation with NO labels),
a state-of-the-art self-supervised learning method.

Main components:
- Models: Backbone architectures and projection heads
- Data: Multi-crop augmentation and dataset utilities
- Loss: DINO loss with centering and temperature scaling
- Training: Trainer class with EMA updates and checkpointing
- Utils: Checkpoint management, logging, and EMA utilities
"""

__version__ = "0.1.0"

from .config import DinoConfig
from .models import get_backbone, DinoProjectionHead, DinoModel
from .loss import DinoLoss
from .training import DinoTrainer

__all__ = [
    'DinoConfig',
    'get_backbone',
    'DinoProjectionHead',
    'DinoModel',
    'DinoLoss',
    'DinoTrainer',
]
