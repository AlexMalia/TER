# Models

This guide covers the neural network architectures used in DINO.

---

## Architecture Overview

DINO uses a student-teacher setup where both networks share the same architecture:

```
DinoModel = Backbone + Projection Head
```

```
Input Image
    ↓
Backbone (ResNet/ViT)
    ↓
Feature Vector [batch, embed_dim]
    ↓
Projection Head (MLP)
    ↓
Output [batch, output_dim]
```

---

## Backbones

### Available Backbones

| Backbone | Source | Output Dim | Parameters |
|----------|--------|------------|------------|
| `resnet18` | torchvision | 512 | 11M |
| `resnet34` | torchvision | 512 | 21M |
| `resnet50` | torchvision | 2048 | 23M |
| `resnet101` | torchvision | 2048 | 42M |
| `resnet152` | torchvision | 2048 | 58M |
| `dino_vits8` | HuggingFace | 384 | 21M |
| `dino_vits16` | HuggingFace | 384 | 21M |
| `dino_vitb8` | HuggingFace | 768 | 85M |
| `dino_vitb16` | HuggingFace | 768 | 85M |

### Using Backbones

```python
from dino.models import get_backbone

# ResNet backbone
backbone = get_backbone('resnet18', pretrained=True)
print(backbone.output_dim)  # 512

# ViT backbone (from HuggingFace)
backbone = get_backbone('dino_vits16', pretrained=True)
print(backbone.output_dim)  # 384
```

### ResNet Backbones

ResNet backbones use torchvision models with the final FC layer removed:

```python
class ResnetBackboneDino(BackboneBase):
    def __init__(self, variant='resnet18', pretrained=False):
        self.model = resnet_variant(pretrained=pretrained)
        self.model.fc = nn.Identity()  # Remove classifier
        self.output_dim = 512  # or 2048 for ResNet50+
```

### ViT Backbones

ViT backbones use pre-trained DINO models from HuggingFace:

```python
class DinoBackbone(BackboneBase):
    VARIANT_MAP = {
        "dino_vits8": "facebook/dino-vits8",
        "dino_vits16": "facebook/dino-vits16",
        "dino_vitb8": "facebook/dino-vitb8",
        "dino_vitb16": "facebook/dino-vitb16",
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x, interpolate_pos_encoding=True)
        return outputs.last_hidden_state[:, 0]  # CLS token
```

---

## Projection Head

The projection head is a backbone-agnostic MLP that projects features to the output space.

### Architecture

```
Input [embed_dim]
    ↓
Linear(embed_dim → hidden_dim) + GELU
    ↓
Linear(hidden_dim → hidden_dim) + GELU
    ↓
Linear(hidden_dim → bottleneck_dim)  # Bottleneck
    ↓
L2 Normalize
    ↓
Weight-Normalized Linear(bottleneck_dim → output_dim)
    ↓
Output [output_dim]
```

### Configuration

```yaml
model:
  projection_hidden_dim: 1024
  projection_bottleneck_dim: 256
  projection_output_dim: 2048
  projection_use_bn: false
```

### Using Projection Head

```python
from dino.models import DinoProjectionHead

# Manual creation
projection = DinoProjectionHead(
    input_dim=512,      # From backbone
    hidden_dim=1024,
    bottleneck_dim=256,
    output_dim=2048,
    use_bn=False
)

# From config
projection = DinoProjectionHead.from_config(config.model, input_dim=512)
```

### Why Weight Normalization?

- Stabilizes training by constraining weight scale
- Prevents gradient explosion in final layer
- Used in the original DINO paper

---

## Complete DINO Model

### Creating a Model

```python
from dino.models import DinoModel

# Using factory method (recommended)
model = DinoModel.from_config(config)

# Manual creation
from dino.models import get_backbone, DinoProjectionHead

backbone = get_backbone('resnet18')
projection = DinoProjectionHead(input_dim=512, output_dim=2048)
model = DinoModel(backbone, projection)
```

### Forward Pass

```python
# Standard forward
projections = model(images)  # [batch, output_dim]

# Get backbone features (for evaluation)
features, projections = model(images, return_backbone_features=True)
```

### Model Properties

```python
model = DinoModel.from_config(config)
print(model.output_dim)  # 2048
print(model.backbone.output_dim)  # 512 (for ResNet18)
```

---

## Student-Teacher Setup

In DINO, two identical models are used:

```python
# Create student and teacher
student = DinoModel.from_config(config)
teacher = DinoModel.from_config(config)

# Initialize teacher with student weights
teacher.load_state_dict(student.state_dict())

# Disable gradients for teacher (updated via EMA)
for param in teacher.parameters():
    param.requires_grad = False
```

The teacher is updated using Exponential Moving Average (EMA):

```python
from dino.utils import update_teacher_EMA

# After each training step
update_teacher_EMA(student, teacher, momentum=0.996)
```

See [Training](training.md) for details on the training loop.

---

## Model Configuration

```yaml
model:
  # Backbone
  backbone: resnet18           # or dino_vits16, resnet50, etc.
  pretrained_backbone: false   # Use pretrained weights

  # Projection head
  projection_hidden_dim: 1024
  projection_bottleneck_dim: 256
  projection_output_dim: 2048
  projection_use_bn: false
```

---

## Adding Custom Backbones

1. **Create backbone class** in `src/dino/models/backbone/`:

```python
# src/dino/models/backbone/efficientnet.py
from .backbone import BackboneBase
import timm

class EfficientNetBackbone(BackboneBase):
    def __init__(self, variant='efficientnet_b0', pretrained=False):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=pretrained, num_classes=0)
        self.output_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)
```

2. **Update factory function** in `src/dino/models/backbone/backbone.py`:

```python
def get_backbone(name, pretrained=False):
    # ... existing code ...
    elif name.startswith('efficientnet'):
        from .efficientnet import EfficientNetBackbone
        return EfficientNetBackbone(name, pretrained)
```

3. **Export in `__init__.py`** and use in config!

---

## See Also

- [Training](training.md) - Training loop and EMA
- [Loss Function](loss-function.md) - DINO loss computation
- [API Reference: Models](../api/models.md) - API documentation
