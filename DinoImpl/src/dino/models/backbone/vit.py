from .backbone import BackboneBase
from transformers import ViTModel, ViTConfig
import torch


class DinoBackbone(BackboneBase):
    """
    DINO (v1) backbone using HuggingFace transformers.

    Uses the CLS token from the last hidden state as output.

    Args:
        variant: DINO variant ('dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16')
        pretrained: Whether to use pre-trained weights
    """

    VARIANT_MAP = {
        "dino_vits8": "facebook/dino-vits8",
        "dino_vits16": "facebook/dino-vits16",
        "dino_vitb8": "facebook/dino-vitb8",
        "dino_vitb16": "facebook/dino-vitb16",
    }

    def __init__(self, variant: str = "dino_vits16", pretrained: bool = False):
        super().__init__()

        if variant not in self.VARIANT_MAP:
            raise ValueError(
                f"Unknown DINO variant: {variant}. "
                f"Available: {list(self.VARIANT_MAP.keys())}"
            )

        model_name = self.VARIANT_MAP[variant]

        if pretrained:
            self.model = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.model = ViTModel(config)

        self.output_dim = self.model.config.hidden_size
        self.variant = variant

        # TODO si timm, mettre num_class self.model.config.hidden_size à verifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Input tensor of shape [batch_size, 3, height, width]

        Returns:
            Feature tensor of shape [batch_size, output_dim]
        """
        outputs = self.model(x, interpolate_pos_encoding=True)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token

    def __repr__(self) -> str:
        return f"DinoBackbone(variant={self.variant}, output_dim={self.output_dim})"