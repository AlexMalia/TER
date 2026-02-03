"""Projection head implementations for DINO."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dino.config import ModelConfig


class DinoProjectionHead(nn.Module):
    """
    Projection head for DINO as described in the paper.

    This is a backbone-agnostic MLP that works with any feature extractor
    (ResNet, ViT, etc.).

    Architecture:
        - 3-layer MLP with GELU activations
        - L2 normalization after bottleneck
        - Weight-normalized final layer (optional)

    Args:
        input_dim: Input dimension (backbone output dimension)
        output_dim: Output dimension (default: 2048 as in paper)
        hidden_dim: Hidden layer dimension (default: 1024)
        bottleneck_dim: Bottleneck dimension before final layer (default: 256)
        use_weight_norm: Whether to use weight normalization on final layer
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 2048,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        use_weight_norm: bool = True
    ):
        super().__init__()

        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("All dimensions must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )

        # Final layer (with optional weight normalization)
        last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)

        if use_weight_norm:
            self.last_layer = torch.nn.utils.parametrizations.weight_norm(last_layer)
        else:
            self.last_layer = last_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection head."""
        # Pass through MLP
        x = self.mlp(x)
        
        # L2 normalize after bottleneck (as per DINO paper)
        x = F.normalize(x, dim=-1, p=2)
        
        # Final projection (weight-normalized layer)
        x = self.last_layer(x)
        
        return x

    @classmethod
    def from_config(cls, model_config: ModelConfig, input_dim: int) -> DinoProjectionHead:
        """
        Factory function to create a DinoProjectionHead from a ModelConfig.

        Args:
            model_config: Model configuration dataclass
            input_dim: Input dimension (backbone output dimension)

        Returns:
            Configured DinoProjectionHead instance
        """
        return cls(
            input_dim=input_dim,
            output_dim=model_config.projection_output_dim,
            hidden_dim=model_config.projection_hidden_dim,
            bottleneck_dim=model_config.projection_bottleneck_dim,
            use_weight_norm=model_config.use_weight_norm
        )

    def __repr__(self) -> str:
        return (
            f"DinoProjectionHead("
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"bottleneck_dim={self.bottleneck_dim})"
        )
