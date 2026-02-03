# Design Decisions

This document explains the key architectural decisions in this DINO implementation.

---

## Design Philosophy

This implementation follows modern Python best practices:

- **Modular design**: Each component has a single, well-defined responsibility
- **Type safety**: Type hints throughout for better IDE support and error catching
- **Configuration over code**: All hyperparameters in YAML, no magic numbers
- **Separation of concerns**: Data, models, training, and utilities are isolated
- **Production ready**: Proper logging, error handling, and checkpointing

---

## Key Decisions

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

**Example**:
```python
@dataclass
class DataConfig:
    dataset: str = 'imagenette'
    batch_size: int = 32
    num_workers: int = 4
```

---

### 2. Why Separate Backbone and Projection?

**Alternative**: Single monolithic model

**Chosen: Composition (DinoModel = Backbone + Projection)**

**Reasons**:
- Backbone features useful for downstream tasks
- Can replace backbone without changing projection logic
- Can freeze backbone for linear evaluation
- Clear separation of feature extraction vs. projection
- Matches DINO paper architecture

**Implementation**:
```python
class DinoModel(nn.Module):
    def __init__(self, backbone, projection_head):
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x, return_backbone_features=False):
        features = self.backbone(x)
        projections = self.projection_head(features)
        if return_backbone_features:
            return features, projections
        return projections
```

---

### 3. Why Custom Collate Function?

**Problem**: Multi-crop transform returns list of tensors

**Default collate behavior**:
```python
batch = [(views_list, label), ...]
default_collate(batch)  # Fails! Can't stack lists
```

**Solution**: Custom collate transposes data structure
```python
# Input: [[g1_img1, g2_img1, l1_img1, ...], [g1_img2, ...], ...]
# Output: [[g1_img1, g1_img2, ...], [g2_img1, g2_img2, ...], ...]

def collate_multi_crop(batch):
    views_lists = [item[0] for item in batch]
    num_views = len(views_lists[0])
    views_batch = [
        torch.stack([views[i] for views in views_lists])
        for i in range(num_views)
    ]
    labels = torch.tensor([item[1] for item in batch])
    return views_batch, labels
```

---

### 4. Why EMA Instead of Two Optimizers?

**Alternative**: Train two models with different optimizers

**Chosen: One student + EMA teacher**

**Reasons**:
- Teacher needs to be stable (slow-changing)
- EMA provides smooth, consistent updates
- Prevents teacher from diverging
- Computationally efficient (no backprop through teacher)
- Matches DINO paper

**Implementation**:
```python
@torch.no_grad()
def update_teacher_EMA(student, teacher, momentum=0.996):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data = momentum * t_param.data + (1 - momentum) * s_param.data
```

---

### 5. Why Checkpoint Both Student and Teacher?

**Alternative**: Only save student (can reconstruct teacher with EMA)

**Chosen: Save both**

**Reasons**:
- Teacher state is not deterministic from student alone
- EMA accumulation history is important
- Ensures exact resumption of training
- Minimal storage overhead (weights are shared structure)

---

### 6. Why Factory Methods (`from_config`)?

**Alternative**: Manual construction everywhere

**Chosen: Factory methods for all major components**

**Benefits**:
- ~90 lines reduced to ~30 lines in train.py
- Correct wiring: Dimensions automatically matched
- Consistency: Same pattern across components
- Testability: Factory methods can be unit tested

**Pattern**:
```python
# Factory method
model = DinoModel.from_config(config)

# vs. manual (still works)
backbone = get_backbone('resnet18')
projection = DinoProjectionHead(input_dim=512, output_dim=2048)
model = DinoModel(backbone, projection)
```

---

### 7. Why Save Loss Center in Checkpoints?

The DINO loss maintains a running center via EMA:

```python
center = momentum * center + (1 - momentum) * teacher_outputs.mean()
```

**Why save it?**
- Center accumulates over many iterations
- Cannot be reconstructed from a single batch
- Critical for stable loss computation
- Required for correct training resumption

---

### 8. Why torchvision for ResNet, HuggingFace for ViT?

**ResNet via torchvision**:
- Standard, well-tested implementations
- Easy to remove classifier head
- Built into PyTorch ecosystem

**ViT via HuggingFace**:
- Pre-trained DINO ViT models available
- Handles position encoding interpolation
- Active community and updates

**Alternative considered**: timm for both
- Pro: Single dependency
- Con: Different API patterns, less common for ViT

---

## Module Responsibilities

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `config` | Configuration management | `DinoConfig`, `DataConfig` |
| `data` | Data pipeline | `DINOTransform`, `get_dataset` |
| `models` | Neural networks | `DinoModel`, `get_backbone` |
| `loss` | Loss computation | `DinoLoss` |
| `training` | Training loop | `DinoTrainer` |
| `utils` | Cross-cutting concerns | `save_checkpoint`, `History` |

---

## Technology Stack

- **PyTorch 2.0+**: Core deep learning framework
- **torchvision**: Pre-trained models and transforms
- **HuggingFace transformers**: ViT backbones
- **uv**: Fast, modern Python package manager
- **YAML**: Human-readable configuration format
- **Python dataclasses**: Type-safe configuration objects
- **tqdm**: Progress bars for training loops

---

## Trade-offs Made

| Decision | Benefit | Cost |
|----------|---------|------|
| Separate backbone/projection | Flexibility, clarity | Slightly more code |
| Dataclasses for config | Type safety, IDE support | Learning curve |
| Custom collate | Correct multi-crop handling | Extra code |
| Save teacher weights | Exact resumption | Larger checkpoints |
| Factory methods | Less boilerplate | Additional abstraction |

---

## What We Don't Do

- **No backward compatibility with old checkpoints**: Simpler code
- **No automatic hyperparameter search**: Out of scope
- **No distributed training built-in**: Use PyTorch's DDP directly
- **No early stopping**: Users can implement if needed

---

## See Also

- [Extending DINO](extending.md) - Adding custom components
- [Performance](performance.md) - Optimization guide
- [Configuration](../guides/configuration.md) - Full config reference
