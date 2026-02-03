# Loss API Reference

API documentation for the loss module.

---

## DinoLoss

::: dino.loss.DinoLoss
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward
        - from_config

---

## Usage Examples

### Creating Loss from Config

```python
from dino.config import DinoConfig
from dino.loss import DinoLoss
from dino.models import DinoModel

config = DinoConfig.from_yaml('configs/default.yaml')
model = DinoModel.from_config(config)

loss_fn = DinoLoss.from_config(
    config.loss,
    config.augmentation,
    out_dim=model.output_dim
)
```

### Manual Loss Creation

```python
from dino.loss import DinoLoss

loss_fn = DinoLoss(
    out_dim=2048,
    student_temp=0.1,
    teacher_temp=0.04,
    center_momentum=0.9,
    ncrops=8,
    n_global_crops=2
)
```

### Computing Loss

```python
import torch

# Student processes all views
student_outputs = []
for view in all_views:  # 8 views
    student_outputs.append(student(view))
student_output = torch.cat(student_outputs, dim=0)

# Teacher processes only global views
with torch.no_grad():
    teacher_outputs = []
    for view in global_views:  # 2 views
        teacher_outputs.append(teacher(view))
    teacher_output = torch.cat(teacher_outputs, dim=0)

# Compute loss
loss = loss_fn(student_output, teacher_output)
loss.backward()
```

### Accessing the Center

The loss maintains a running center for stability:

```python
# Get current center
center = loss_fn.center

# The center is updated automatically during forward()
# It's also saved/loaded with checkpoints
```

### Loss Configuration

```yaml
loss:
  student_temp: 0.1        # Higher = softer distribution
  teacher_temp: 0.04       # Lower = sharper distribution
  center_momentum: 0.9     # EMA momentum for centering
```
