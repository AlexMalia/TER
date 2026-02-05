# Configuration

Complete guide to the YAML configuration system.

---

## Overview

All settings are managed through YAML configuration files using Python dataclasses for type safety.

```yaml
# configs/default.yaml
data:
  dataset: imagenette
  batch_size: 32

model:
  backbone: resnet18

training:
  num_epochs: 100
```

---

## Configuration Hierarchy

```
DinoConfig
├── DataConfig           # Dataset, batch size, splits
├── AugmentationConfig   # Crop sizes, color jitter, etc.
├── ModelConfig          # Backbone, projection dims
├── LossConfig           # Temperatures, center momentum
├── OptimizerConfig      # Learning rate, weight decay
├── SchedulerConfig      # LR scheduler configuration
├── TrainingConfig       # Epochs, teacher momentum
├── CheckpointConfig     # Save frequency, directory
└── LoggingConfig        # Log directory, TensorBoard
```

---

## Loading Configuration

### From YAML

```python
from dino.config import DinoConfig

config = DinoConfig.from_yaml('configs/default.yaml')
```

### CLI Overrides

```python
# Load YAML, then override
config = DinoConfig.from_yaml('config.yaml')
if args.batch_size:
    config.data.batch_size = args.batch_size
if args.lr:
    config.optimizer.lr = args.lr
```

**Priority**: CLI args > YAML file > Defaults

---

## Configuration Sections

### DataConfig

```yaml
data:
  dataset: imagenette         # Dataset name (imagenette, imagenet100)
  data_path: ./data           # Root path for datasets
  batch_size: 32              # Training batch size
  num_workers: 4              # DataLoader workers
  pin_memory: true            # Pin memory for GPU
  train_split: 0.9            # Train/val split ratio
```

### AugmentationConfig

```yaml
augmentation:
  # Crop settings
  n_global_crops: 2           # Number of global crops
  num_local_views: 6          # Number of local crops
  global_crop_size: 224       # Global crop size (pixels)
  local_crop_size: 96         # Local crop size (pixels)
  global_crop_scale: [0.4, 1.0]   # Scale range for global crops
  local_crop_scale: [0.05, 0.4]   # Scale range for local crops

  # Color augmentation
  color_jitter_prob: 0.8      # Probability of color jitter
  brightness: 0.4
  contrast: 0.4
  saturation: 0.2
  hue: 0.1

  # Other augmentations
  gaussian_blur_prob: 0.5     # Probability of Gaussian blur
  solarization_prob: 0.2      # Probability of solarization

  # Normalization (ImageNet defaults)
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

### ModelConfig

```yaml
model:
  # Backbone
  backbone: resnet18          # resnet18/34/50/101/152, dino_vits8/16, dino_vitb8/16
  pretrained_backbone: false  # Use pretrained weights

  # Projection head
  projection_hidden_dim: 1024
  projection_bottleneck_dim: 256
  projection_output_dim: 2048
  use_weight_norm: true       # Weight normalization on final layer
```

### LossConfig

```yaml
loss:
  student_temp: 0.1           # Student temperature (higher = softer)
  teacher_temp: 0.04          # Teacher temperature (lower = sharper)
  center_momentum: 0.9        # EMA momentum for centering
```

### OptimizerConfig

```yaml
optimizer:
  optimizer: adamw            # Optimizer type
  lr: 0.001                   # Learning rate
  weight_decay: 0.04          # Weight decay
  betas: [0.9, 0.999]         # Adam betas
```

### SchedulerConfig

```yaml
scheduler:
  scheduler: cosine_warmup    # Scheduler type
  warmup_epochs: 10           # Warmup epochs
  min_lr: 1.0e-6              # Minimum learning rate
  warmup_start_lr: 0.0        # Starting LR for warmup
```

### TrainingConfig

```yaml
training:
  num_epochs: 100             # Total epochs
  teacher_momentum: 0.996     # EMA momentum for teacher
  teacher_momentum_final: 1.0 # Final momentum (if scheduled)
  teacher_momentum_schedule: true # Use momentum scheduling
  gradient_clip: 3.0          # Gradient clipping (null to disable)
```

### CheckpointConfig

```yaml
checkpoint:
  save_dir: ./checkpoints     # Checkpoint directory
  save_freq: 10               # Save every N epochs
  save_latest: true           # Keep latest checkpoint
  save_best: true             # Keep best checkpoint
```

### LoggingConfig

```yaml
logging:
  log_dir: ./logs             # Log directory
  log_freq: 100               # Log every N iterations
  use_tensorboard: true       # Enable TensorBoard
  log_level: INFO             # Logging level
```

---

## Complete Example

```yaml
# configs/imagenet100.yaml
data:
  dataset: imagenet100
  data_path: ./data/imagenet100
  batch_size: 64
  num_workers: 8
  pin_memory: true

augmentation:
  n_global_crops: 2
  num_local_views: 6
  global_crop_size: 224
  local_crop_size: 96

model:
  backbone: resnet50
  pretrained_backbone: false
  projection_output_dim: 2048

loss:
  student_temp: 0.1
  teacher_temp: 0.04
  center_momentum: 0.9

optimizer:
  optimizer: adamw
  lr: 0.0005
  weight_decay: 0.04

scheduler:
  scheduler: cosine_warmup
  warmup_epochs: 10
  min_lr: 1.0e-6

training:
  num_epochs: 200
  teacher_momentum: 0.996
  gradient_clip: 3.0

checkpoint:
  save_dir: ./checkpoints
  save_freq: 10

logging:
  log_dir: ./logs
  use_tensorboard: true
```

---

## Validation

Configuration is validated at multiple levels:

1. **Type checking**: Dataclass enforces types
2. **Value checking**: In `__post_init__` methods
3. **Runtime checking**: In component constructors

Example validation:

```python
@dataclass
class LossConfig:
    student_temp: float = 0.1
    teacher_temp: float = 0.04

    def __post_init__(self):
        if self.student_temp <= 0:
            raise ValueError("Temperature must be positive")
        if self.student_temp <= self.teacher_temp:
            warnings.warn("Student temp should be > teacher temp")
```

---

## Using Factory Methods

Components can be created directly from config:

```python
# Models
model = DinoModel.from_config(config)

# Loss
loss_fn = DinoLoss.from_config(config.loss, config.augmentation, out_dim)

# Transforms
transform = DINOTransform.from_config(config.augmentation)

# Optimizer and Scheduler
optimizer = create_optimizer(model.parameters(), config.optimizer)
scheduler = create_scheduler(optimizer, config.scheduler, config.optimizer, total_steps, warmup_steps)
```

---

## Factory Methods Pattern

Components can be created directly from config using `from_config()` methods. This reduces boilerplate and ensures correct parameter wiring.

### Available Factory Methods

| Component | Factory Method | Purpose |
|-----------|---------------|---------|
| `DinoModel` | `from_config(config)` | Creates backbone + projection with correct dimensions |
| `DinoProjectionHead` | `from_config(model_config, input_dim)` | Creates projection head from model config |
| `DinoLoss` | `from_config(loss_config, aug_config, out_dim)` | Creates loss with correct crop counts |
| `DINOTransform` | `from_config(aug_config)` | Creates multi-crop transform |

### Benefits

1. **Less boilerplate**: Complex initialization reduced to single lines
2. **Correct wiring**: Dimensions and parameters automatically matched
3. **Consistency**: Same pattern across all components
4. **Testability**: Factory methods can be unit tested independently

### Example: Manual vs Factory Creation

**Manual creation** (verbose but explicit):
```python
from dino.models import get_backbone, DinoProjectionHead, DinoModel

backbone = get_backbone('resnet18', pretrained=False)
projection = DinoProjectionHead(
    input_dim=backbone.output_dim,  # Must match backbone output!
    hidden_dim=1024,
    bottleneck_dim=256,
    output_dim=2048,
    use_weight_norm=True
)
model = DinoModel(backbone, projection)
```

**Factory creation** (recommended):
```python
from dino.models import DinoModel

model = DinoModel.from_config(config)  # Handles everything automatically
```

The factory method internally:
1. Creates the backbone based on `config.model.backbone`
2. Reads `backbone.output_dim` to wire the projection head correctly
3. Applies all projection head parameters from `config.model`

---

## See Also

- [CLI Reference](../getting-started/cli-reference.md) - Command-line options
- [Training](training.md) - Training configuration details
- [API Reference](../api/models.md) - Component API documentation
