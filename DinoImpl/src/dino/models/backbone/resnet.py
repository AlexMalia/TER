
from .backbone import BackboneBase
from torchvision import models
import torch
import torch.nn as nn


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

        print(f"Initialized {variant} backbone with output_dim={self.output_dim}, pretrained={pretrained}")


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
