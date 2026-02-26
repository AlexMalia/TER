"""DINO model wrapper combining backbone and projection head."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple
from dino.config import ModelConfig


class DinoModel(nn.Module):
    """
    Complete DINO model combining backbone and projection head.

    This model can be used for both student and teacher networks.

    Args:
        backbone: Backbone network (e.g., ResNet)
        projection_head: Projection head network
    """

    def __init__(self, backbone: nn.Module, projection_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.output_dim = projection_head.output_dim

    def forward(
        self,
        x: torch.Tensor,
        return_backbone_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            return_backbone_features: If True, return both backbone features
                                       and projections. Useful for evaluation.

        Returns:
            If return_backbone_features is False:
                Projected features of shape [batch_size, output_dim]
            If return_backbone_features is True:
                Tuple of (backbone_features, projections)
                    backbone_features: [batch_size, backbone_dim]
                    projections: [batch_size, output_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Project features
        projections = self.projection_head(features)

        if return_backbone_features:
            return features, projections
        return projections

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract only backbone features without projection.

        Useful for evaluation and transfer learning.

        Args:
            x: Input tensor of shape [batch_size, 3, height, width]

        Returns:
            Backbone features of shape [batch_size, backbone_dim]
        """
        return self.backbone(x)

    def parameter_count(self) -> dict:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts:
                - total: Total parameters
                - trainable: Trainable parameters
                - backbone: Backbone parameters
                - projection: Projection head parameters
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        projection = sum(p.numel() for p in self.projection_head.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'backbone': backbone,
            'projection': projection
        }

    @classmethod
    def from_config(cls, model_config: ModelConfig) -> DinoModel:
        """
        Create a DinoModel from a ModelConfig.

        Args:
            model_config: Model configuration dataclass

        Returns:
            Configured DinoModel instance
        """
        from .backbone import get_backbone
        from .projection_head import DinoProjectionHead

        backbone = get_backbone(
            model_config.backbone,
            pretrained=model_config.is_backbone_pretrained
        )
        projection_head = DinoProjectionHead.from_config(
            model_config,
            input_dim=backbone.output_dim
        )
        return cls(backbone, projection_head)

    def __repr__(self) -> str:
        param_counts = self.parameter_count()
        return (
            f"DinoModel(\n"
            f"  backbone={self.backbone.__class__.__name__},\n"
            f"  projection_head={self.projection_head.__class__.__name__},\n"
            f"  total_params={param_counts['total']:,},\n"
            f"  output_dim={self.output_dim}\n"
            f")"
        )
