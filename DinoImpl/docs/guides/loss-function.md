# Loss Function

This guide explains the DINO loss function, including temperature scaling, centering, and the cross-entropy computation.

---

## Overview

The DINO loss encourages the student to match the teacher's predictions across different views of the same image.

```
Student (all 8 views) → Softmax → ─────┐
                                       ├→ Cross-Entropy Loss
Teacher (2 global views) → Softmax → ──┘
```

---

## Core Computation

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
            if i == j: continue  # Skip same view
            loss += -sum(teacher_view * student_view)

    # 5. Update center with EMA
    center = momentum * center + (1 - momentum) * teacher_outputs.mean()

    return loss / num_pairs
```

---

## Key Mechanisms

### Temperature Scaling

Temperature controls the "sharpness" of the probability distribution:

- **Lower temperature** → More peaked distribution (more confident)
- **Higher temperature** → Flatter distribution (less confident)

| Component | Temperature | Effect |
|-----------|-------------|--------|
| Student | τ=0.1 | Sharper, more confident |
| Teacher | τ=0.04 | Even sharper, provides strong guidance |

```python
# Student: temperature = 0.1
student_probs = softmax(logits / 0.1)

# Teacher: temperature = 0.04 (sharper)
teacher_probs = softmax(logits / 0.04)
```

### Centering

Centering prevents collapse (all predictions being the same):

```python
# Center is running mean of teacher outputs
center = momentum * center + (1 - momentum) * teacher_outputs.mean()

# Subtract center before temperature scaling
teacher_logits = (teacher_outputs - center) / teacher_temp
```

**Why centering?**

- Without centering, the model can collapse to predicting the same thing for all inputs
- Centering ensures the teacher outputs are balanced around zero
- EMA update provides smooth, stable centering

### Asymmetry

The key asymmetry in DINO:

- **Teacher**: Only sees **global views** (2 large crops)
- **Student**: Sees **all views** (2 global + 6 local crops)

This forces the student to learn features that work across scales and augmentations.

---

## Using DinoLoss

### From Config (Recommended)

```python
from dino.loss import DinoLoss

loss_fn = DinoLoss.from_config(
    config.loss,
    config.augmentation,
    out_dim=model.output_dim
)
```

### Manual Creation

```python
loss_fn = DinoLoss(
    out_dim=2048,
    student_temp=0.1,
    teacher_temp=0.04,
    center_momentum=0.9,
    ncrops=8,
    n_global_crops=2
)
```

### Forward Pass

```python
# During training
student_output = torch.cat([student(v) for v in all_views])
teacher_output = torch.cat([teacher(v) for v in global_views])

loss = loss_fn(student_output, teacher_output)
loss.backward()
```

---

## Configuration

```yaml
loss:
  student_temp: 0.1
  teacher_temp: 0.04
  center_momentum: 0.9

augmentation:
  n_global_crops: 2
  num_local_views: 6
```

---

## Important: The Bug Fix

The original notebook had a bug that caused negative loss values. The issue was iterating over raw tensors instead of chunked probability distributions.

**Wrong (original)**:
```python
for i, teacher_output in enumerate(teacher_outputs):  # Raw tensors!
    for j, student_output in enumerate(student_outputs):
        loss = -torch.sum(teacher_output * student_output)
```

**Correct (fixed)**:
```python
# Chunk outputs into views
student_views = student_log_probs.chunk(ncrops)
teacher_views = teacher_probs.chunk(n_global_crops)

for i, teacher_prob in enumerate(teacher_views):  # Chunked distributions
    for j, student_log_prob in enumerate(student_views):
        loss = -torch.sum(teacher_prob * student_log_prob)
```

This implementation includes the fix - loss values should always be positive.

---

## Mathematical Details

### Cross-Entropy

For each (teacher view, student view) pair:

$$L_{i,j} = -\sum_k P_t^{(i)}(k) \log P_s^{(j)}(k)$$

Where:
- $P_t^{(i)}$ = teacher probability distribution for view $i$
- $P_s^{(j)}$ = student probability distribution for view $j$

### Total Loss

Sum over all valid pairs (excluding same view):

$$L = \frac{1}{N} \sum_{i \in \{global\}} \sum_{j \in \{all\}, j \neq i} L_{i,j}$$

Where $N$ is the number of valid pairs.

---

## Troubleshooting

### Loss is NaN

- Check learning rate (try reducing it)
- Verify temperature values are positive
- Check for extreme values in outputs

### Loss is Negative

If you see negative loss values, something is wrong with the loss computation. This implementation should always produce positive loss values.

### Loss Not Decreasing

- Check batch size (need sufficient examples for meaningful statistics)
- Verify data augmentation is working
- Try adjusting temperature values

---

## See Also

- [Training](training.md) - Training loop and optimization
- [Models](models.md) - Network architectures
- [API Reference: Loss](../api/loss.md) - API documentation
