from .backbone import BackboneBase
import timm
import torch


class DinoBackbone(BackboneBase):
    """
    DINO (v1) backbone using timm.

    Uses the CLS token from the last hidden state as output.

    Args:
        variant: DINO variant ('dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16')
        pretrained: Whether to use pre-trained weights
    """

    VARIANT_MAP = {
        "dino_vits8": "vit_small_patch8_224",
        "dino_vits16": "vit_small_patch16_224",
        "dino_vitb8": "vit_base_patch8_224",
        "dino_vitb16": "vit_base_patch16_224",
    }

    def __init__(self, variant: str = "dino_vits16", pretrained: bool = False):
        super().__init__()

        if variant not in self.VARIANT_MAP:
            raise ValueError(
                f"Unknown DINO variant: {variant}. "
                f"Available: {list(self.VARIANT_MAP.keys())}"
            )

        model_name = self.VARIANT_MAP[variant]

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # remove classification head, forward returns CLS token
            dynamic_img_size=True, # Interpolate small crop 
        )

        self.output_dim = self.model.num_features
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
        return f"DinoBackbone(variant={self.variant}, output_dim={self.output_dim})"