# Checkpointing

Guide to saving, loading, and resuming training from checkpoints.

---

## Overview

Checkpoints save the complete training state, allowing you to:

- Resume training after interruption
- Evaluate models at specific epochs
- Share trained models

---

## Checkpoint Contents

Each checkpoint includes:

```python
checkpoint = {
    'epoch': int,                    # Current epoch
    'iteration': int,                # Current iteration
    'student_state_dict': OrderedDict,   # Student weights
    'teacher_state_dict': OrderedDict,   # Teacher weights
    'optimizer_state_dict': dict,    # Optimizer state
    'dino_loss_center': Tensor,      # Loss centering buffer
    'config': dict,                  # Training configuration
    'metrics': dict,                 # Training metrics
    'timestamp': str                 # Save time
}
```

### Why Save the Loss Center?

The DINO loss maintains a running center via EMA. This center is critical for:

- Stable loss computation
- Preventing collapse
- Consistent training resumption

Losing the center would break loss computation when resuming.

---

## Automatic Saving

Checkpoints are saved automatically during training:

```yaml
checkpoint:
  save_dir: ./checkpoints     # Where to save
  save_freq: 10               # Save every N epochs
  save_latest: true           # Keep checkpoint_latest.pth
  save_best: true             # Keep checkpoint_best.pth
```

### Files Created

- `checkpoint_epoch_XXXX.pth` - Checkpoint for specific epoch
- `checkpoint_latest.pth` - Most recent checkpoint
- `checkpoint_best.pth` - Best performing checkpoint (if enabled)

---

## Resuming Training

### From Command Line

```bash
# Resume from latest
python scripts/train.py --resume checkpoints/checkpoint_latest.pth

# Resume from specific epoch
python scripts/train.py --resume checkpoints/checkpoint_epoch_0050.pth
```

### From Python

```python
from dino.utils import load_checkpoint

# Load checkpoint
checkpoint = load_checkpoint(
    'checkpoints/checkpoint_latest.pth',
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    loss_fn=loss_fn
)

# Get resume information
start_epoch = checkpoint['epoch'] + 1
iteration = checkpoint['iteration']
```

---

## Manual Saving

```python
from dino.utils import save_checkpoint

save_checkpoint(
    path='checkpoints/my_checkpoint.pth',
    epoch=50,
    iteration=10000,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    loss_fn=loss_fn,
    config=config,
    metrics={'loss': 2.5}
)
```

---

## Loading for Evaluation

When loading for evaluation (not training), you only need the model weights:

```python
import torch
from dino.models import DinoModel

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_best.pth')

# Create model
model = DinoModel.from_config(config)

# Load weights (student or teacher)
model.load_state_dict(checkpoint['student_state_dict'])
# or for evaluation, teacher often performs better:
model.load_state_dict(checkpoint['teacher_state_dict'])

model.eval()
```

---

## Extracting Features

```python
# Load trained model
checkpoint = torch.load('checkpoints/checkpoint_best.pth')
model = DinoModel.from_config(config)
model.load_state_dict(checkpoint['teacher_state_dict'])
model.eval()

# Extract features
with torch.no_grad():
    # Get backbone features (for downstream tasks)
    features, projections = model(images, return_backbone_features=True)

    # Features shape: [batch, backbone_dim] (e.g., 512 for ResNet18)
    # Projections shape: [batch, projection_dim] (e.g., 2048)
```

---

## Checkpoint Compatibility

!!! warning "Version Compatibility"
    Checkpoints are not guaranteed to be compatible across different code versions. If you've made significant changes to model architecture, you may need to retrain from scratch.

### Common Issues

**Missing keys**:
```python
# Strict loading (fails on mismatch)
model.load_state_dict(checkpoint['state_dict'], strict=True)

# Lenient loading (ignores mismatches)
model.load_state_dict(checkpoint['state_dict'], strict=False)
```

**Architecture changes**:
- If you change backbone or projection dimensions, old checkpoints won't work
- Keep track of which config was used for each checkpoint

---

## Best Practices

1. **Always save config**: Include the configuration used for training
2. **Version your checkpoints**: Use descriptive names or directories
3. **Save both student and teacher**: Teacher often performs better for evaluation
4. **Save regularly**: Prevent loss from crashes or interruptions
5. **Keep best checkpoint**: Enable `save_best: true` in config

---

## Configuration

```yaml
checkpoint:
  save_dir: ./checkpoints     # Checkpoint directory
  save_freq: 10               # Save every N epochs
  save_latest: true           # Always keep latest
  save_best: true             # Keep best performing
```

---

## See Also

- [Training](training.md) - Training loop details
- [CLI Reference](../getting-started/cli-reference.md) - Resume options
- [API Reference: Utils](../api/utils.md) - Checkpoint functions
