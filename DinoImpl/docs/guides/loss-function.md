# Loss Function

This guide explains the DINO loss function, including temperature scaling, centering, and the cross-entropy computation.

---

## Overview

The DINO loss encourages the student to match the teacher's predictions across different views of the same image.

```
Student (all 8 views) → Softmax → ─────┐
                                       ├> Cross-Entropy Loss
Teacher (2 global views) → Softmax → ──┘
```

---

## Core Computation

```python
def forward(self, student_outputs, teacher_outputs):
    # 1. Temperature scaling
    student_logits = student_outputs / student_temp  # Softer
    teacher_logits = (teacher_outputs - center) / teacher_temp  # Sharper

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
    center = center * center_momentum + batch_center * (1 - momentum) * batch_center

    return loss / num_pairs
```

---

## Key Mechanisms

### Temperature Scaling

Temperature controls the "sharpness" of the probability distribution:

- **Lower temperature** : More peaked distribution (more confident)
- **Higher temperature** : Flatter distribution (less confident)

| Component | Temperature | Effect |
|-----------|-------------|--------|
| Student | 0.1 | Slightly softer, allows learning flexibility |
| Teacher | 0.04 | Sharper, provides confident target distributions |

**Why this asymmetry?**

The temperature difference serves a crucial pedagogical purpose:

1. **Teacher (0.04, sharper)**: The teacher produces highly peaked, confident probability distributions. These act as strong "pseudo-labels" that tell the student which output dimensions are most important for a given input. A lower temperature amplifies small differences in logits.

2. **Student (0.1, softer)**: The student uses a slightly higher temperature, producing softer distributions. This provides:
   - **Learning flexibility**: The student isn't forced to exactly match the teacher's sharp peaks immediately
   - **Gradient flow**: Softer distributions provide smoother gradients during backpropagation
   - **Exploration**: Small but non-zero probabilities on other dimensions allow the student to explore

This mechanism create a similar effect to knowledge distillation, where the teacher's confident predictions guide the student

Dino original values are 0.1 for the student and 0.04 for the teacher.

### Centering

Centering prevents collapse (all predictions being the same):

```python
# Subtract center before temperature scaling
teacher_logits = (teacher_outputs - center) / teacher_temp
```

```python
# Center is running mean of teacher outputs
batch_center = teacher_output.mean(dim=0, keepdim=True)
center = center * center_momentum + batch_center * (1 - momentum)
```

in the original implementation the center_momentum as a value of 0.9, it means the center is updated 90% with the previous center and 10% with the new batch center (means of the teacher outputs of last batch).

**Why centering?**

Without centering, the model can collapse to a trivial solution where it outputs the same distribution for all inputs. 

**Collapse scenario (From dino's papper)**:

- One dimension dominates (e.g., all outputs are the same one-hot vector)
- Uniform collapse (e.g., all outputs are uniform distributions)

The centering mechanism prevent one dimension to dominate but encourage collapse to the uniform distribution.
The sharpenning mechanism prevent the uniform distribution collapse but encourage one dimension to dominate.

Applying both operations balances their effects which is sufficient to avoid collapse in presence of a momentum teacher

**The centering mechanism**:

$$c \leftarrow m \cdot c + (1 - m) \cdot \mathbb{E}_{x}[g_{\theta_t}(x)]$$

Where:
- $c$ is the center vector (same dimension as output, e.g., 2048)
- $m$ is the center momentum (default 0.9)
- $g_{\theta_t}(x)$ is the teacher output for input $x$
- The expectation is computed over the current batch

**How it prevents collapse**:

- The center tracks the mean teacher output across all recent batches
- Subtracting the center ensures teacher outputs are zero-mean
- If the model tries to collapse (all outputs identical), the center would equal that output, making centered outputs all zero
- Zero centered outputs lead to uniform softmax, which is a high-loss state
- This creates a self-correcting mechanism that pushes the model away from collapse

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

## See Also

- [Training](training.md) - Training loop and optimization
- [Models](models.md) - Network architectures
- [API Reference: Loss](../api/loss.md) - API documentation
