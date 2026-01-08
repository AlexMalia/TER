"""Backbone architectures for DINO."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


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


class ResnetBackboneDino(BackboneBase):
    """
    ResNet backbone for DINO.

    Removes the final fully connected layer and uses the penultimate layer
    features as output.

    Args:
        variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        pretrained: Whether to use pre-trained ImageNet weights
    """

    def __init__(self, variant: str = "resnet18", pretrained: bool = False):
        super().__init__()

        # Get the appropriate ResNet model
        if variant == "resnet18":
            model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        elif variant == "resnet34":
            model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        elif variant == "resnet50":
            model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        elif variant == "resnet101":
            model = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)
        elif variant == "resnet152":
            model = models.resnet152(weights='IMAGENET1K_V2' if pretrained else None)
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")

        # Store output dimension
        self.output_dim = model.fc.in_features

        # Remove the final fully connected layer
        model.fc = nn.Identity()
        self.model = model
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Input tensor of shape [batch_size, 3, height, width]

        Returns:
            Feature tensor of shape [batch_size, output_dim]
        """
        return self.model(x)

    def __repr__(self) -> str:
        return f"ResnetBackboneDino(variant={self.variant}, output_dim={self.output_dim})"


def get_backbone(name: str, pretrained: bool = False, **kwargs) -> BackboneBase:
    """
    Factory function to get backbone by name.

    Args:
        name: Backbone name ('resnet18', 'resnet34', 'resnet50', etc.)
        pretrained: Whether to use pre-trained weights
        **kwargs: Additional arguments for the backbone

    Returns:
        Backbone instance

    Example:
        >>> backbone = get_backbone('resnet18', pretrained=False)
        >>> print(backbone.output_dim)
        512
    """
    name_lower = name.lower()

    if name_lower in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        return ResnetBackboneDino(variant=name_lower, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {name}")
