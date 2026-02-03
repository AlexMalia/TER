"""DINO models package."""

from .backbone import get_backbone, BackboneBase, ResnetBackboneDino, DinoBackbone
from .projection_head import DinoProjectionHead

# Deprecated alias for backward compatibility
CovNetProjectionHeadDino = DinoProjectionHead
from .dino_model import DinoModel

__all__ = [
    'get_backbone',
    'BackboneBase',
    'ResnetBackboneDino',
    'DinoBackbone',
    'DinoProjectionHead',
    'CovNetProjectionHeadDino',  # Deprecated alias
    'DinoModel',
]
