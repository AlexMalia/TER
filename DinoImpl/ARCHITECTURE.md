# DINO Architecture & Implementation Details

This document provides an in-depth technical overview of the DINO implementation, including design decisions, code structure, and extension points.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Code Architecture](#code-architecture)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Configuration System](#configuration-system)
6. [Training Pipeline](#training-pipeline)
7. [Key Design Decisions](#key-design-decisions)
8. [Extension Guide](#extension-guide)
9. [Performance Considerations](#performance-considerations)
10. [Known Issues & Fixes](#known-issues--fixes)

---

## Project Overview

### Design Philosophy

This implementation follows modern Python best practices:
- **Modular design**: Each component has a single, well-defined responsibility
- **Type safety**: Type hints throughout for better IDE support and error catching
- **Configuration over code**: All hyperparameters in YAML, no magic numbers
- **Separation of concerns**: Data, models, training, and utilities are isolated
- **Production ready**: Proper logging, error handling, and checkpointing

### Technology Stack

- **PyTorch 2.0+**: Core deep learning framework
- **torchvision**: Pre-trained models and transforms
- **uv**: Fast, modern Python package manager
- **YAML**: Human-readable configuration format
- **Python dataclasses**: Type-safe configuration objects
- **tqdm**: Progress bars for training loops

---

## Code Architecture

### Package Structure

```
src/dino/
├── __init__.py              # Package exports
├── config/                  # Configuration management
│   ├── config.py           # Dataclass definitions
│   └── __init__.py
├── data/                    # Data loading and augmentation
│   ├── transforms.py       # Multi-crop augmentation
│   ├── datasets.py         # Dataset wrappers
│   ├── dataloaders.py      # DataLoader creation
│   └── __init__.py
├── models/                  # Neural network architectures
│   ├── backbone/           # Backbone architectures
│   │   ├── backbone.py     # Base class and factory function
│   │   ├── resnet.py       # ResNet backbones
│   │   ├── vit.py          # DINO ViT backbones (HuggingFace)
│   │   └── __init__.py
│   ├── projection_head.py  # Projection head (with from_config)
│   ├── dino_model.py       # Complete DINO model (with from_config)
│   └── __init__.py
├── loss/                    # Loss functions
│   ├── dino_loss.py        # DINO cross-entropy loss (with from_config)
│   └── __init__.py
├── training/                # Training orchestration
│   ├── trainer.py          # DinoTrainer class
│   ├── optim.py            # create_optimizer, create_scheduler helpers
│   └── __init__.py
├── evaluation/              # Evaluation utilities
│   └── __init__.py
└── utils/                   # Shared utilities
    ├── ema.py              # Exponential Moving Average
    ├── checkpoint.py       # Checkpoint management
    ├── logging_utils.py    # Logging setup
    ├── history.py          # Training history tracking and visualization
    └── __init__.py
```

### Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|----------------|----------------------|
| `config` | Configuration management | `DinoConfig`, `DataConfig`, `ModelConfig`, `SchedulerConfig` |
| `data` | Data pipeline | `DINOTransform`, `get_dataset`, `create_dataloaders` |
| `models` | Neural networks | `ResnetBackboneDino`, `DinoBackbone` (ViT), `DinoModel.from_config()` |
| `loss` | Loss computation | `DinoLoss`, `DinoLoss.from_config()` |
| `training` | Training loop | `DinoTrainer`, `create_optimizer`, `create_scheduler` |
| `evaluation` | Model evaluation | (future: KNN evaluation, linear probing) |
| `utils` | Cross-cutting concerns | `save_checkpoint`, `update_teacher_EMA`, `History` |

---

## Component Details

### 1. Configuration System (`config/config.py`)

**Design Pattern**: Nested dataclasses with YAML serialization

```python
@dataclass
class DinoConfig:
    data: DataConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
```

**Key Features**:
- Type-safe configuration with validation
- Nested structure mirrors logical grouping
- YAML serialization/deserialization
- Command-line override support

**AugmentationConfig** includes:
```python
@dataclass
class AugmentationConfig:
    n_global_crops: int = 2      # Number of global crops
    num_local_views: int = 6     # Number of local crops
    # ... other fields

    @property
    def ncrops(self) -> int:
        """Total number of crops (global + local)."""
        return self.n_global_crops + self.num_local_views
```

**Factory Method Pattern (`from_yaml`)**: The `DinoConfig.from_yaml()` method uses a registry pattern for cleaner config parsing:
```python
_CONFIG_CLASSES = {
    'data': DataConfig,
    'augmentation': AugmentationConfig,
    'model': ModelConfig,
    'loss': LossConfig,
    'optimizer': OptimizerConfig,
    'scheduler': SchedulerConfig,
    'training': TrainingConfig,
    'checkpoint': CheckpointConfig,
    'logging': LoggingConfig,
}
```

**Why dataclasses?**
- Type hints for IDE autocomplete
- Automatic `__init__` generation
- Built-in equality and repr methods
- No runtime overhead

### 2. Data Pipeline (`data/`)

#### Multi-Crop Transform (`transforms.py`)

**Class**: `DINOTransform`

**Purpose**: Creates multiple augmented views of the same image

**Architecture**:
```python
class DINOTransform:
    def __init__(self, num_local_views=6, ...):
        # Global view 1: Standard augmentation
        self.global_t1 = Compose([
            RandomResizedCrop(224, scale=(0.4, 1.0)),
            ColorJitter + GaussianBlur
        ])

        # Global view 2: Add solarization
        self.global_t2 = Compose([
            ..., RandomSolarize(...)
        ])

        # Local views: Smaller crops
        self.local_transform = Compose([
            RandomResizedCrop(96, scale=(0.05, 0.4)),
            ...
        ])

    def __call__(self, img) -> List[Tensor]:
        return [
            self.global_t1(img),
            self.global_t2(img),
            *[self.local_transform(img) for _ in range(6)]
        ]
```

**Key Design Decisions**:
- Different augmentations for global views (solarization only on 2nd)
- Local crops use more aggressive cropping (scale: 0.05-0.4)
- All parameters configurable via `AugmentationConfig`

#### Dataset Management (`datasets.py`)

**Factory Pattern**: `get_dataset(name, path, transform, ...)`

**Supported Datasets**:
- **ImageNette**: `torchvision.datasets.Imagenette` - 10-class subset of ImageNet
- **ImageNet100**: Custom loader for Kaggle's 100-class ImageNet subset

**ImageNet100 Details**:
- Downloaded from Kaggle: `kaggle datasets download -d ambityga/imagenet100`
- Multi-folder structure: `train.X1`, `train.X2`, `train.X3`, `train.X4` (combined via `ConcatDataset`)
- Validation folder: `val.X`
- ~130,000 training images, 5,000 validation images

**Extension Point**: To add new datasets, add case to `get_dataset()`:
```python
elif dataset_name == 'your_dataset':
    return YourDataset(root=data_path, transform=transform)
```

#### DataLoader Creation (`dataloaders.py`)

**Key Function**: `create_dataloaders(config)`

**Custom Collate Function**:
```python
def collate_multi_crop(batch):
    """Transpose list-of-lists to list of batched tensors"""
    views_lists = [item[0] for item in batch]  # Each: [g1, g2, l1, ..., l6]
    labels = [item[1] for item in batch]

    # Transpose: [[g1_img1, g1_img2, ...], [g2_img1, g2_img2, ...], ...]
    views_batch = [
        torch.stack([views[i] for views in views_lists])
        for i in range(num_views)
    ]
    return views_batch, torch.tensor(labels)
```

**Why custom collate?**
- Multi-crop returns list of tensors, not single tensor
- Need to group by view type (all g1 together, all g2 together, etc.)

### 3. Models (`models/`)

#### Backbone (`backbone/`)

**Base Class**: `BackboneBase` (abstract interface in `backbone.py`)

**Implementations**:

1. **`ResnetBackboneDino`** (`resnet.py`): ResNet variants
   - Variants: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
   - Output dim: 512 (ResNet18/34) or 2048 (ResNet50+)
   - Removes final FC layer, uses penultimate features

2. **`DinoBackbone`** (`vit.py`): DINO ViT variants via HuggingFace transformers
   - Variants: `dino_vits8`, `dino_vits16`, `dino_vitb8`, `dino_vitb16`
   - Output dim: 384 (ViT-S) or 768 (ViT-B)
   - Uses CLS token from last hidden state

**Factory Function**: `get_backbone(name, pretrained=False)`
```python
# ResNet
backbone = get_backbone('resnet18', pretrained=True)

# DINO ViT (from HuggingFace)
backbone = get_backbone('dino_vits16', pretrained=True)
```

**ViT Backbone Details**:
```python
class DinoBackbone(BackboneBase):
    VARIANT_MAP = {
        "dino_vits8": "facebook/dino-vits8",   # Patch size 8, ViT-S
        "dino_vits16": "facebook/dino-vits16", # Patch size 16, ViT-S
        "dino_vitb8": "facebook/dino-vitb8",   # Patch size 8, ViT-B
        "dino_vitb16": "facebook/dino-vitb16", # Patch size 16, ViT-B
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x, interpolate_pos_encoding=True)
        return outputs.last_hidden_state[:, 0]  # CLS token
```

**Key Method**:
```python
def forward(self, x: Tensor) -> Tensor:
    """Extract features from input images"""
    return self.model(x)  # [batch, output_dim]
```

#### Projection Head (`projection_head.py`)

**Class**: `DinoProjectionHead`

This is a **backbone-agnostic MLP** that works identically with any feature extractor (ResNet, ViT, etc.).

**Architecture** (DINO paper):
```
Input (512)
    ↓
Linear(512 → 1024) + GELU
    ↓
Linear(1024 → 1024) + GELU
    ↓
Linear(1024 → 256)  # Bottleneck
    ↓
L2 Normalize
    ↓
Weight-Normalized Linear(256 → 2048)
    ↓
Output (2048)
```

**Factory Method**:
```python
@classmethod
def from_config(cls, model_config: ModelConfig, input_dim: int) -> "DinoProjectionHead":
    """Create projection head from config."""
    return cls(
        input_dim=input_dim,
        hidden_dim=model_config.projection_hidden_dim,
        bottleneck_dim=model_config.projection_bottleneck_dim,
        output_dim=model_config.projection_output_dim,
        use_bn=model_config.projection_use_bn,
    )
```

**Why weight normalization?**
- Stabilizes training by constraining weight scale
- Prevents gradient explosion in final layer
- Used in original DINO paper

#### Complete Model (`dino_model.py`)

**Class**: `DinoModel`

**Composition**:
```python
class DinoModel(nn.Module):
    def __init__(self, backbone, projection_head):
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x, return_backbone_features=False):
        features = self.backbone(x)
        projections = self.projection_head(features)

        if return_backbone_features:
            return features, projections  # For evaluation
        return projections
```

**Factory Method**:
```python
@classmethod
def from_config(cls, config: DinoConfig) -> "DinoModel":
    """Create complete DINO model from config.

    Automatically creates backbone and projection head with correct dimensions.
    """
    backbone = get_backbone(
        config.model.backbone,
        pretrained=config.model.pretrained_backbone
    )
    projection_head = DinoProjectionHead.from_config(
        config.model,
        input_dim=backbone.output_dim
    )
    return cls(backbone, projection_head)
```

**Why separate backbone and projection?**
- Backbone features are useful for downstream tasks
- Projection head only needed during pre-training
- Can freeze backbone and train linear classifier

### 4. Loss Function (`loss/dino_loss.py`)

**Class**: `DinoLoss`

**Factory Method**:
```python
@classmethod
def from_config(
    cls,
    loss_config: LossConfig,
    aug_config: AugmentationConfig,
    out_dim: int
) -> "DinoLoss":
    """Create DinoLoss from config objects."""
    return cls(
        out_dim=out_dim,
        student_temp=loss_config.student_temp,
        teacher_temp=loss_config.teacher_temp,
        center_momentum=loss_config.center_momentum,
        ncrops=aug_config.ncrops,
        n_global_crops=aug_config.n_global_crops,
    )
```

**Core Computation**:
```python
def forward(self, student_outputs, teacher_outputs):
    # 1. Temperature scaling
    student_logits = student_outputs / student_temp  # Sharper
    teacher_logits = (teacher_outputs - center) / teacher_temp  # Softer

    # 2. Probability distributions
    student_log_probs = log_softmax(student_logits)
    teacher_probs = softmax(teacher_logits).detach()  # No gradients

    # 3. Chunk into views
    student_views = student_log_probs.chunk(8)  # All views
    teacher_views = teacher_probs.chunk(2)      # Global only

    # 4. Cross-entropy between all pairs (except same view)
    for i, teacher_view in enumerate(teacher_views):
        for j, student_view in enumerate(student_views):
            if i == j: continue
            loss += -sum(teacher_view * student_view)

    # 5. Update center with EMA
    center = momentum * center + (1 - momentum) * teacher_outputs.mean()

    return loss / num_pairs
```

**Key Mechanisms**:

1. **Temperature Scaling**:
   - Student: τ=0.1 (sharper, more confident)
   - Teacher: τ=0.04 (even sharper, guiding)
   - Lower temperature = more peaked distribution

2. **Centering**:
   - Subtracts running mean of teacher outputs
   - Prevents collapse (all predictions same)
   - EMA update: α=0.9

3. **Asymmetry**:
   - Teacher only sees global views
   - Student sees all views (global + local)
   - Forces student to learn from limited teacher info

**Bug Fix**: Original notebook iterated over raw tensors instead of chunked distributions!

### 5. Training Pipeline (`training/trainer.py`)

**Class**: `DinoTrainer`

**Training Loop Structure**:
```python
for epoch in range(num_epochs):
    for batch_idx, (views, _) in enumerate(train_loader):
        # 1. Forward pass
        global_views, all_views = get_global_local_views(views)

        teacher_out = concat([teacher(v) for v in global_views])
        student_out = concat([student(v) for v in all_views])

        # 2. Loss computation
        loss = dino_loss(student_out, teacher_out)

        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()
        if gradient_clip: clip_grad_norm(student.parameters())
        optimizer.step()

        # 4. EMA update for teacher
        momentum = schedule[iter] if scheduled else base_momentum
        update_teacher_EMA(student, teacher, momentum)

        # 5. Logging and checkpointing
        if iter % log_freq == 0: log_metrics(...)
        if iter % checkpoint_freq == 0: save_checkpoint(...)
```

**Design Features**:
- Progress bar with tqdm
- Configurable gradient clipping
- Optional momentum scheduling
- Periodic checkpointing
- Clean separation of concerns

### 6. Utilities (`utils/`)

#### EMA Updates (`ema.py`)

**Core Function**:
```python
@torch.no_grad()
def update_teacher_EMA(student, teacher, alpha=0.99):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data
```

**Momentum Scheduling**:
```python
def get_momentum_schedule(base, final, num_epochs, niter_per_epoch):
    """Cosine schedule from base (0.996) to final (1.0)"""
    schedule = []
    for progress in range(total_iterations):
        momentum = final - (final - base) * (cos(π * progress / total) + 1) / 2
        schedule.append(momentum)
    return schedule
```

**Why schedule momentum?**
- Early training: Lower momentum (0.996), teacher learns faster
- Late training: Higher momentum (→1.0), teacher becomes very stable
- Improves final performance

#### Checkpointing (`checkpoint.py`)

**Checkpoint Contents**:
```python
checkpoint = {
    'epoch': int,
    'iteration': int,
    'student_state_dict': OrderedDict,
    'teacher_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'dino_loss_center': Tensor,  # Important for stability!
    'config': dict,
    'metrics': dict,
    'timestamp': str
}
```

**Why save loss center?**
- Center is accumulated via EMA during training
- Losing it would break loss computation
- Critical for training continuity

#### History Tracking (`history.py`)

**Class**: `History`

**Purpose**: Track and visualize training metrics over time

**Core Features**:
- Records metrics at iteration and epoch granularity
- JSON serialization/deserialization
- Matplotlib plotting for loss, learning rate, and momentum
- Pandas DataFrame export

**API**:
```python
# Create and record
history = History(metadata={'config': config.to_dict()})
history.record_iteration(iter, {'loss': 2.5, 'learning_rate': 0.001, 'momentum': 0.996})
history.record_epoch(epoch, {'loss': 2.3, 'learning_rate': 0.001, 'momentum': 0.997})

# Save/Load
history.save('training_history.json')
history = History.load('training_history.json')

# Access metrics
losses = history.get_metric('loss', level='epoch')
iterations = history.get_iterations()

# Plotting
history.plot_loss(level='epoch')
history.plot_learning_rate(level='iteration')
history.plot_momentum(level='iteration')
history.plot_all(level='epoch', save_path='plots.png')

# Export
df = history.to_dataframe(level='epoch')  # Requires pandas
```

**Data Structure**:
```python
{
    'metadata': {'created_at': '...', 'config': {...}},
    'iteration_metrics': [
        {'iteration': 0, 'timestamp': '...', 'loss': 2.5, 'learning_rate': 0.001, 'momentum': 0.996},
        ...
    ],
    'epoch_metrics': [
        {'epoch': 1, 'timestamp': '...', 'loss': 2.3, 'learning_rate': 0.001, 'momentum': 0.997},
        ...
    ]
}
```

---

## Data Flow

### End-to-End Pipeline

```
Image Dataset
    ↓
DINOTransform (multi-crop augmentation)
    ↓
DataLoader (batch + collate)
    ↓
[8 views per image: 2 global + 6 local]
    ↓
├─→ Teacher Network (global views only)
│   └→ [batch×2, 2048]
│
└─→ Student Network (all views)
    └→ [batch×8, 2048]
    ↓
DinoLoss (cross-entropy + centering)
    ↓
Backprop through student only
    ↓
EMA update: teacher ← momentum * teacher + (1-momentum) * student
    ↓
Repeat
```

### Tensor Shapes Through Pipeline

| Stage | Shape | Notes |
|-------|-------|-------|
| Original image | `[3, H, W]` | Single RGB image |
| After DINOTransform | `List[8 tensors]` | Each: `[3, crop_size, crop_size]` |
| After collate | `List[8 batches]` | Each: `[batch, 3, crop_size, crop_size]` |
| After backbone | `[batch, 512]` | Feature vectors |
| After projection | `[batch, 2048]` | Projected embeddings |
| Teacher output | `[batch×2, 2048]` | Concatenated global views |
| Student output | `[batch×8, 2048]` | Concatenated all views |
| Loss | `scalar` | Single value to minimize |

---

## Configuration System

### Hierarchy

```
DinoConfig
├── DataConfig           # Dataset, batch size, splits
├── AugmentationConfig  # Crop sizes, color jitter, etc.
├── ModelConfig         # Backbone, projection dims
├── LossConfig          # Temperatures, center momentum
├── OptimizerConfig     # Learning rate, weight decay
├── SchedulerConfig     # Learning rate scheduler configuration
├── TrainingConfig      # Epochs, teacher momentum
├── CheckpointConfig    # Save frequency, directory
└── LoggingConfig       # Log directory, TensorBoard
```

### SchedulerConfig Details

The `SchedulerConfig` dataclass configures learning rate scheduling:

```python
@dataclass
class SchedulerConfig:
    scheduler: str = "cosine_warmup"   # Scheduler type
    warmup_epochs: int = 10            # Number of warmup epochs
    min_lr: float = 1e-6               # Minimum learning rate
    warmup_start_lr: float = 0.0       # Starting LR for warmup
```

**Scheduler Types**:
- `cosine_warmup`: Cosine annealing with linear warmup (recommended)

**How it works**:
1. Linear warmup from `warmup_start_lr` to `optimizer.lr` over `warmup_epochs`
2. Cosine decay from `optimizer.lr` to `min_lr` for remaining epochs

### Override Mechanism

**Priority**: CLI args > YAML file > Defaults

```bash
# Load YAML, then override
python scripts/train.py --config custom.yaml --batch-size 64 --lr 0.0005
```

**Implementation**:
```python
config = DinoConfig.from_yaml(args.config)
if args.batch_size: config.data.batch_size = args.batch_size
if args.lr: config.optimizer.lr = args.lr
```

### Validation

Validation happens at multiple levels:
1. **Type checking**: Dataclass enforces types
2. **Value checking**: In `__post_init__` methods
3. **Runtime checking**: In component constructors

Example:
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

## Training Pipeline

### Initialization Sequence

The initialization has been simplified using factory methods:

1. **Configuration**: Load YAML → Override with CLI args
2. **Logging**: Setup file and console handlers
3. **Data**: Create dataloaders with `create_dataloaders(config)`
4. **Models**: Create with `DinoModel.from_config(config)` (handles backbone + projection)
5. **Teacher**: Copy student weights → Disable gradients
6. **Loss**: Create with `DinoLoss.from_config(config.loss, config.augmentation, out_dim)`
7. **Optimizer**: Create with `create_optimizer(params, config.optimizer)`
8. **Scheduler**: Create with `create_scheduler(optimizer, config.scheduler, config.optimizer, total_steps, warmup_steps)`
9. **Trainer**: Assemble all components
10. **Resume** (optional): Load checkpoint
11. **Train**: Run training loop

**Simplified Example** (see `scripts/train.py`):
```python
# Load config
config = DinoConfig.from_yaml('config.yaml')

# Create all components with factory methods
train_loader, val_loader, _ = create_dataloaders(config)
student = DinoModel.from_config(config)
teacher = DinoModel.from_config(config)
teacher.load_state_dict(student.state_dict())

loss_fn = DinoLoss.from_config(config.loss, config.augmentation, student.output_dim)
optimizer = create_optimizer(student.parameters(), config.optimizer)
scheduler = create_scheduler(optimizer, config.scheduler, config.optimizer, total_steps, warmup_steps)
```

### Training Loop Anatomy

```python
# Outer loop: Epochs
for epoch in range(start_epoch, num_epochs):
    # Inner loop: Batches
    for batch_idx, (views, labels) in enumerate(train_loader):
        # 1. Prepare views
        global_views = views[:2]     # First 2 are global
        all_views = views            # All 8 views

        # 2. Teacher forward (no gradients)
        with torch.no_grad():
            teacher_outputs = [teacher(v) for v in global_views]
            teacher_output = torch.cat(teacher_outputs, dim=0)

        # 3. Student forward (with gradients)
        student_outputs = [student(v) for v in all_views]
        student_output = torch.cat(student_outputs, dim=0)

        # 4. Compute loss
        loss = dino_loss(student_output, teacher_output)

        # 5. Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 6. Gradient clipping (optional)
        if config.training.gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                config.training.gradient_clip
            )

        # 7. Optimizer step
        optimizer.step()

        # 8. Learning rate scheduler step (per iteration)
        if scheduler is not None:
            scheduler.step()

        # 9. EMA update for teacher
        momentum = get_current_momentum()
        update_teacher_EMA(student, teacher, momentum)

        # 10. Logging and history tracking
        if should_log(batch_idx):
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
            history.record_iteration(iteration, {
                'loss': loss.item(),
                'learning_rate': current_lr,
                'momentum': momentum
            })

        # 11. Checkpointing
        if should_save(batch_idx):
            save_checkpoint(...)

    # End of epoch
    history.record_epoch(epoch, {...})
    log_epoch_metrics()
    save_checkpoint()
```

---

## Factory Methods Pattern

The codebase uses a consistent `from_config()` pattern across components to simplify initialization. This reduces boilerplate in training scripts and ensures correct parameter wiring.

### Overview

| Component | Factory Method | Purpose |
|-----------|---------------|---------|
| `DinoModel` | `from_config(config)` | Creates backbone + projection with correct dimensions |
| `DinoProjectionHead` | `from_config(model_config, input_dim)` | Creates projection head from model config |
| `DinoLoss` | `from_config(loss_config, aug_config, out_dim)` | Creates loss with correct crop counts |
| `DINOTransform` | `from_config(aug_config)` | Creates multi-crop transform |

### Optimizer & Scheduler Helpers

The `training/optim.py` module provides helper functions:

```python
from dino.training import create_optimizer, create_scheduler

# Create optimizer from config
optimizer = create_optimizer(model.parameters(), config.optimizer)

# Create scheduler (handles warmup + cosine decay)
scheduler = create_scheduler(
    optimizer,
    config.scheduler,
    config.optimizer,
    total_steps=total_steps,
    warmup_steps=warmup_steps
)
```

### Benefits

1. **Less boilerplate**: ~90 lines reduced to ~30 lines in train.py
2. **Correct wiring**: Dimensions and parameters automatically matched
3. **Consistency**: Same pattern across all components
4. **Testability**: Factory methods can be unit tested independently

### Manual Creation Still Supported

The factory methods are an **alternative** - manual creation remains valid:

```python
# Manual (still works)
backbone = get_backbone('resnet18')
projection = DinoProjectionHead(input_dim=512, output_dim=2048)
model = DinoModel(backbone, projection)

# Factory method (simpler)
model = DinoModel.from_config(config)
```

---

## Key Design Decisions

### 1. Why Dataclasses for Configuration?

**Alternatives considered**:
- Dict-based configs (e.g., `config['data']['batch_size']`)
- Argparse-only (CLI args)
- Hydra framework

**Chosen: Dataclasses + YAML**

**Reasons**:
- Type safety and IDE autocomplete
- No runtime dependencies (Hydra is heavy)
- Easy serialization with YAML
- Hierarchical structure matches logical grouping
- Simple to test and validate

### 2. Why Separate Backbone and Projection?

**Alternative**: Single monolithic model

**Chosen: Composition (DinoModel = Backbone + Projection)**

**Reasons**:
- Backbone features useful for downstream tasks
- Can replace backbone without changing projection logic
- Can freeze backbone for linear evaluation
- Clear separation of feature extraction vs. projection
- Matches DINO paper architecture

### 3. Why Custom Collate Function?

**Problem**: Multi-crop transform returns list of tensors

**Default collate behavior**:
```python
batch = [(views_list, label), ...]
default_collate(batch)  # Fails! Can't stack lists of different lengths
```

**Solution**: Custom collate transposes data structure
```python
# Input: [[g1_img1, g2_img1, l1_img1, ...], [g1_img2, g2_img2, ...], ...]
# Output: [[g1_img1, g1_img2, ...], [g2_img1, g2_img2, ...], ...]
```

### 4. Why EMA Instead of Two Optimizers?

**Alternative**: Train two models with different optimizers

**Chosen: One student + EMA teacher**

**Reasons**:
- Teacher needs to be stable (slow-changing)
- EMA provides smooth, consistent updates
- Prevents teacher from diverging
- Computationally efficient (no backprop through teacher)
- Matches DINO paper

### 5. Why Checkpoint Both Student and Teacher?

**Alternative**: Only save student (can reconstruct teacher with EMA)

**Chosen: Save both**

**Reasons**:
- Teacher state is not deterministic from student alone
- EMA accumulation history is important
- Ensures exact resumption of training
- Minimal storage overhead (weights are shared structure)

---

## Extension Guide

### Adding a New Backbone

The codebase already includes two backbone implementations as examples:

1. **ResNet** (`src/dino/models/backbone/resnet.py`): Uses torchvision
2. **DINO ViT** (`src/dino/models/backbone/vit.py`): Uses HuggingFace transformers

To add a new backbone:

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

3. **Export in `__init__.py`** and update config!

### Adding a New Dataset

1. **Add to `get_dataset()`** in `src/dino/data/datasets.py`:
```python
elif dataset_name == 'custom_dataset':
    return torchvision.datasets.ImageFolder(
        root=os.path.join(data_path, 'custom'),
        transform=transform
    )
```

2. **Create config file** `configs/custom_dataset.yaml`

3. **Run training**:
```bash
python scripts/train.py --config configs/custom_dataset.yaml
```

### Adding Evaluation Metrics

1. **Create evaluator** in `src/dino/evaluation/`:
```python
class KNNEvaluator:
    def __init__(self, model, k=20):
        self.model = model
        self.k = k

    def evaluate(self, train_loader, val_loader):
        # Extract features
        train_features, train_labels = self.extract_features(train_loader)
        val_features, val_labels = self.extract_features(val_loader)

        # Fit KNN
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(train_features, train_labels)

        # Evaluate
        accuracy = knn.score(val_features, val_labels)
        return {'knn_acc': accuracy}
```

2. **Add to trainer**:
```python
class DinoTrainer:
    def validate_epoch(self, epoch):
        if self.evaluator:
            metrics = self.evaluator.evaluate(...)
            log_metrics(metrics, epoch, prefix="Val")
```

### Adding Learning Rate Scheduling

1. **Create scheduler** in `src/dino/training/schedulers.py`:
```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

2. **Integrate in trainer**:
```python
self.scheduler = get_cosine_schedule_with_warmup(...)
# In training loop:
self.scheduler.step()
```

---

## Performance Considerations

### Memory Optimization

**Multi-crop memory usage**:
- 8 views × batch size = 8× memory vs. single view
- Use smaller batch size (16-32 instead of 64-128)
- Consider gradient accumulation:
```python
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Training Speed

**Bottlenecks**:
1. Data loading: Use `num_workers=8`, `pin_memory=True`
2. Augmentation: Multi-crop is expensive, consider caching
3. Forward passes: Use mixed precision (FP16)

**Mixed precision example**:
```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Training

**Future work**: Add DDP support
```python
# Wrap models
student = DDP(student, device_ids=[local_rank])
# Note: Teacher doesn't need DDP (no backprop)

# Synchronize center in loss
if dist.is_available() and dist.is_initialized():
    dist.all_reduce(batch_center)
    batch_center /= dist.get_world_size()
```

---

## Known Issues & Fixes

### Issue 1: Negative Loss Values (FIXED)

**Symptom**: Loss values are negative and increasing in magnitude

**Root cause**: Original notebook bug - iterated over raw outputs instead of probability distributions

**Fix**: Changed loop to use chunked distributions:
```python
# WRONG (original):
for i, teacher_output in enumerate(teacher_outputs):  # Raw tensors!
    for j, student_output in enumerate(student_outputs):
        loss = -torch.sum(teacher_output * student_output)

# CORRECT (fixed):
for i, teacher_prob in enumerate(teacher_probs):  # Chunked distributions
    for j, student_log_prob in enumerate(student_log_probs):
        loss = -torch.sum(teacher_prob * student_log_prob)
```

**Location**: `src/dino/loss/dino_loss.py:88-96`

### Issue 2: YAML Tuple Serialization

**Symptom**: Tuples in config become lists after YAML round-trip

**Workaround**: Use lists in YAML, convert to tuples in dataclass if needed

```yaml
# YAML (must use list syntax)
normalize_mean: [0.485, 0.456, 0.406]

# Dataclass (can annotate as tuple)
normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
```

### Issue 3: Checkpoint Compatibility

**Problem**: Checkpoints from old notebook don't work with new code

**Solution**: Don't support backward compatibility - retrain from scratch

**Reasoning**: Codebase was refactored significantly, maintaining compatibility would add complexity

---

## Advanced Features

### Weights & Biases Integration

The training pipeline integrates with [Weights & Biases](https://wandb.ai) for experiment tracking:

```yaml
logging:
  use_wandb: true
  wandb_project: dino-training
  wandb_entity: your-username
  wandb_run_name: experiment-1
```

**Features**:
- Automatic logging of iteration and epoch metrics
- Configuration saved as W&B config
- Resume support: `wandb_run_id` is saved in checkpoints
- Tags for backbone and dataset

### Gradient Accumulation

For training with limited GPU memory while simulating larger batch sizes:

```yaml
training:
  gradient_accumulation_steps: 4  # Effective batch = batch_size × 4
```

**Implementation details**:
- Loss is divided by `accumulation_steps` before backward pass
- Optimizer step only occurs every N mini-batches
- Scheduler steps are counted as optimizer updates, not forward passes
- EMA update happens after each optimizer step

### Kaggle Training Support

Ready-to-use configurations for Kaggle GPU notebooks:

- `configs/kaggle-imagenette.yaml`: Quick experiments on ImageNette
- `configs/kaggle-imagenet100.yaml`: Full training on ImageNet100

**Key adaptations**:
- Data paths point to `/kaggle/input/` and `/kaggle/working/`
- Reduced workers (2) for Kaggle container limits
- W&B integration with Kaggle secrets support

---

## Testing Strategy

### Unit Tests (TODO)

```python
# tests/test_transforms.py
def test_dino_transform_output_shape():
    transform = DINOTransform(num_local_views=6)
    img = PIL.Image.new('RGB', (256, 256))
    views = transform(img)
    assert len(views) == 8  # 2 global + 6 local
    assert views[0].shape == (3, 224, 224)  # Global
    assert views[2].shape == (3, 96, 96)    # Local

# tests/test_loss.py
def test_dino_loss_positive():
    loss_fn = DinoLoss(out_dim=128)
    student_out = torch.randn(64, 128)  # 8 views × 8 batch
    teacher_out = torch.randn(16, 128)  # 2 views × 8 batch
    loss = loss_fn(student_out, teacher_out)
    assert loss.item() > 0  # Loss must be positive!

# tests/test_checkpoint.py
def test_checkpoint_resume():
    # Save checkpoint
    save_checkpoint(student, teacher, optimizer, loss_fn, ...)

    # Create new models
    student2, teacher2 = create_models()

    # Load checkpoint
    load_checkpoint('checkpoint.pth', student2, teacher2, ...)

    # Verify weights match
    assert torch.allclose(student.parameters(), student2.parameters())
```

### Integration Tests

```bash
# Quick training test (1 epoch, small dataset)
python scripts/train.py --config configs/test.yaml --epochs 1
```

---

## References

### Papers

1. **DINO (Original)**: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021
2. **MoCo**: He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020
3. **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020

### Code References

- Official DINO: https://github.com/facebookresearch/dino
- PyTorch Examples: https://github.com/pytorch/examples
- HuggingFace Transformers (for ViT backbones): https://github.com/huggingface/transformers
- Timm: https://github.com/huggingface/pytorch-image-models

---

**Last Updated**: February 2026

For questions or improvements, open an issue or PR!
