# Models API Reference

API documentation for the models module.

---

## DinoModel

::: dino.models.DinoModel
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward
        - from_config
        - output_dim

---

## Backbones

### get_backbone

::: dino.models.get_backbone
    options:
      show_root_heading: true
      show_source: true

### BackboneBase

::: dino.models.backbone.BackboneBase
    options:
      show_root_heading: true
      show_source: true

### ResnetBackboneDino

::: dino.models.backbone.ResnetBackboneDino
    options:
      show_root_heading: true
      show_source: true

### DinoBackbone (ViT)

::: dino.models.backbone.DinoBackbone
    options:
      show_root_heading: true
      show_source: true

---

## Projection Head

### DinoProjectionHead

::: dino.models.DinoProjectionHead
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward
        - from_config

---

## Usage Examples

### Creating a Model from Config

```python
from dino.config import DinoConfig
from dino.models import DinoModel

config = DinoConfig.from_yaml('configs/default.yaml')
model = DinoModel.from_config(config)

print(f"Output dimension: {model.output_dim}")
```

### Manual Model Creation

```python
from dino.models import get_backbone, DinoProjectionHead, DinoModel

# Create components
backbone = get_backbone('resnet18', pretrained=True)
projection = DinoProjectionHead(
    input_dim=backbone.output_dim,
    hidden_dim=1024,
    bottleneck_dim=256,
    output_dim=2048
)
model = DinoModel(backbone, projection)
```

### Using Different Backbones

```python
from dino.models import get_backbone

# ResNet variants
resnet18 = get_backbone('resnet18')
resnet50 = get_backbone('resnet50', pretrained=True)

# ViT variants
vit_small = get_backbone('dino_vits16', pretrained=True)
vit_base = get_backbone('dino_vitb16', pretrained=True)
```

### Forward Pass

```python
# Standard forward
projections = model(images)  # [batch, output_dim]

# Get backbone features for evaluation
features, projections = model(images, return_backbone_features=True)
# features: [batch, backbone_dim]
# projections: [batch, output_dim]
```
