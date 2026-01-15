"""Backbone architectures for DINO."""

import torch
import torch.nn as nn


class BackboneBase(nn.Module):
    """Base class for backbone architectures."""

    def __init__(self):
        super().__init__()
        self.output_dim = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """Get the output dimension of the backbone."""
        return self.output_dim



def get_backbone(name: str, pretrained: bool = False, **kwargs) -> BackboneBase:
    """
    Factory function to get backbone by name.

    Args:
        name: Backbone name ('resnet18', 'resnet50', 'dinov2_small', 'dinov2_base', etc.)
        pretrained: Whether to use pre-trained weights
        **kwargs: Additional arguments for the backbone

    Returns:
        Backbone instance
    """
    # Import here to avoid circular imports
    from .resnet import ResnetBackboneDino
    from .vit import DinoBackbone

    name_lower = name.lower()

    # ResNet variants
    if name_lower in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        return ResnetBackboneDino(variant=name_lower, pretrained=pretrained)

    # DINO v1
    elif name_lower in ['dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16']:
        return DinoBackbone(variant=name_lower, pretrained=pretrained)

    else:
        raise ValueError(
            f"Unknown backbone: {name}. "
            f"Available: resnet18/34/50/101/152, dino_vits8/vits16/vitb8/vitb16"
        )