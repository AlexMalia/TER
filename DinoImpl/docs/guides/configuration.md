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
  train_split: 0.7            # Training split ratio
  val_split: 0.15             # Validation split ratio
  seed: 42                    # Random seed for reproducibility
```

### AugmentationConfig

```yaml
augmentation:
  # Crop settings
  n_global_crops: 2               # Number of global crops
  num_local_views: 6              # Number of local crops
  global_crop_size: 224           # Global crop size (pixels)
  local_crop_size: 96             # Local crop size (pixels)
  global_crop_scale_min: 0.4      # Min scale for global crops
  global_crop_scale_max: 1.0      # Max scale for global crops
  local_crop_scale_min: 0.05      # Min scale for local crops
  local_crop_scale_max: 0.4       # Max scale for local crops

  # Color augmentation
  color_jitter_prob: 0.8          # Probability of color jitter
  color_jitter_brightness: 0.4
  color_jitter_contrast: 0.4
  color_jitter_saturation: 0.2
  color_jitter_hue: 0.1

  # Other augmentations
  horizontal_flip_prob: 0.5       # Probability of horizontal flip
  grayscale_prob: 0.2             # Probability of grayscale
  gaussian_blur_sigma_min: 0.1    # Gaussian blur sigma range
  gaussian_blur_sigma_max: 2.0
  solarization_prob: 0.2          # Probability of solarization
  solarization_threshold: 128     # Solarization threshold

  # Normalization (ImageNet defaults)
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

### ModelConfig

```yaml
model:
  # Backbone
  backbone: resnet18          # resnet18/34/50/101/152, dino_vits8/16, dino_vitb8/16
  backbone_pretrained: false  # Use pretrained weights

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
  scheduler: cosine_warmup    # Scheduler type (cosine_warmup)
  warmup_epochs: 10           # Number of warmup epochs
  min_lr: 1.0e-6              # Minimum learning rate after decay
  warmup_start_lr: 0.0        # Starting LR for warmup (ramps up to optimizer.lr)
```

### TrainingConfig

```yaml
training:
  num_epochs: 100             # Total epochs
  teacher_momentum: 0.996     # EMA momentum for teacher
  teacher_momentum_final: 1.0 # Final momentum (if scheduled)
  teacher_momentum_schedule: true # Use momentum scheduling
  gradient_clip: 3.0          # Gradient clipping (null to disable)
  gradient_accumulation_steps: 1  # Accumulate gradients for larger effective batch
  mixed_precision: false      # Use mixed precision training (experimental)
  seed: 42                    # Random seed
  device: cuda                # Device to use (cuda, cpu)
```

### CheckpointConfig

```yaml
checkpoint:
  checkpoint_dir: ./checkpoints   # Checkpoint directory
  save_every_n_epochs: 1          # Save every N epochs
  save_every_n_iters: null        # Save every N iterations (optional)
  keep_last_n: 5                  # Number of checkpoints to keep
  save_best: true                 # Keep best checkpoint
  resume_from: null               # Path to checkpoint to resume from
```

### LoggingConfig

```yaml
logging:
  log_dir: ./logs             # Log directory
  log_every_n_iters: 10       # Log every N iterations
  log_verbosity: info         # Logging level (debug, info, warning, error)

  # Weights & Biases integration
  use_wandb: false            # Enable W&B experiment tracking
  wandb_project: null         # W&B project name
  wandb_entity: null          # W&B entity (username or team)
  wandb_run_name: null        # Custom run name (optional)
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
  train_split: 0.85
  val_split: 0.15
  seed: 42

augmentation:
  n_global_crops: 2
  num_local_views: 6
  global_crop_size: 224
  local_crop_size: 96

model:
  backbone: resnet50
  backbone_pretrained: false
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
  teacher_momentum_schedule: true
  teacher_momentum_final: 1.0
  gradient_clip: 3.0
  gradient_accumulation_steps: 1
  seed: 42

checkpoint:
  checkpoint_dir: ./checkpoints
  save_every_n_epochs: 10
  save_best: true

logging:
  log_dir: ./logs
  log_every_n_iters: 50
  log_verbosity: info
  use_wandb: false
```

---

## Kaggle Configuration

For training on Kaggle, use the specialized configuration files:

```yaml
# configs/kaggle-imagenet100.yaml
data:
  dataset: imagenet100
  data_path: /kaggle/input/imagenet100
  batch_size: 32
  num_workers: 2

training:
  gradient_accumulation_steps: 2  # Simulate larger batch size

checkpoint:
  checkpoint_dir: /kaggle/working/checkpoints

logging:
  log_dir: /kaggle/working/logs
  use_wandb: true
  wandb_project: dino-training
```

See [Kaggle Training](#kaggle-training) for more details.

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
