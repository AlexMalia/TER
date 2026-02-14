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
| `dino_vits8` | Timm | 384 | 21M |
| `dino_vits16` | Timm | 384 | 21M |
| `dino_vitb8` | Timm | 768 | 85M |
| `dino_vitb16` | Timm | 768 | 85M |

### Using Backbones

Manually create a backbone (Not recommended - use factory method for consistency):
```python
from dino.models import get_backbone

# ResNet backbone
backbone = get_backbone('resnet18', pretrained=True)
print(backbone.output_dim)  # 512

# ViT backbone (from Timm)
backbone = get_backbone('dino_vits16', pretrained=True)
print(backbone.output_dim)  # 384
```

From DinoModel factory method:
```python
from dino.models import DinoModel

model = DinoModel.from_config(config.model_config)
backbone = model.backbone
```

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
  projection_output_dim: 4096
  use_weight_norm: true
```

Dino original values :
- `projection_hidden_dim`: 2048
- `projection_bottleneck_dim`: 256
- `projection_output_dim`: 65536

From dino's paper :

|  (Output dim) | 1024 | 4096 | 16384 | **65536** | 262144 |
| --- | --- | --- | --- | --- | --- |
| **KNN top-1** | 67.8 | 69.3 | 69.2 | **69.7** | 69.1 |


### Using Projection Head

```python
from dino.models import DinoProjectionHead

# Manual creation (Not recommended - use factory method for consistency)
projection = DinoProjectionHead(
    input_dim=512,      # From backbone
    hidden_dim=1024,
    bottleneck_dim=256,
    output_dim=4096,
    use_weight_norm=True
)

# From config
student = DinoModel.from_config(config.model_config)
projection = student.projection_head
```

### Why Weight Normalization?

The projection head uses weight normalization on the final layer for training stability:

- **L2 normalization after bottleneck**: Features are normalized before the final linear layer, ensuring consistent scale regardless of input magnitude. This normalization helps with deep projection heads.
- **Weight normalization on final layer**: Decouples the direction and magnitude of weights, preventing gradient explosion
- **Why it matters**: Without these techniques, the self-supervised training can become unstable, especially with high-dimensional output spaces (2048 dimensions)

The original DINO paper found this combination crucial for preventing mode collapse and maintaining stable gradients during the knowledge distillation process

---

## Complete DINO Model

### Creating a Model

```python
from dino.models import DinoModel

# Using factory method (recommended)
model = DinoModel.from_config(config)
backbone = model.backbone
projection = model.projection_head

# Manual creation (Not recommended - use factory method for consistency)
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
