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
| Student | τ=0.1 | Slightly softer, allows learning flexibility |
| Teacher | τ=0.04 | Sharper, provides confident target distributions |

```python
# Student: temperature = 0.1
student_probs = softmax(logits / 0.1)

# Teacher: temperature = 0.04 (sharper)
teacher_probs = softmax(logits / 0.04)
```

**Why this asymmetry?**

The temperature difference serves a crucial pedagogical purpose:

1. **Teacher (τ=0.04, sharper)**: The teacher produces highly peaked, confident probability distributions. These act as strong "pseudo-labels" that tell the student which output dimensions are most important for a given input. A lower temperature amplifies small differences in logits.

2. **Student (τ=0.1, softer)**: The student uses a slightly higher temperature, producing softer distributions. This provides:
   - **Learning flexibility**: The student isn't forced to exactly match the teacher's sharp peaks immediately
   - **Gradient flow**: Softer distributions provide smoother gradients during backpropagation
   - **Exploration**: Small but non-zero probabilities on other dimensions allow the student to explore

The ratio (0.1/0.04 = 2.5×) creates a "soft labeling" effect similar to knowledge distillation, where the teacher's confident predictions guide the student without being overly prescriptive

### Centering

Centering prevents collapse (all predictions being the same):

```python
# Center is running mean of teacher outputs
center = momentum * center + (1 - momentum) * teacher_outputs.mean()

# Subtract center before temperature scaling
teacher_logits = (teacher_outputs - center) / teacher_temp
```

**Why centering?**

Without centering, the model can collapse to a trivial solution where it outputs the same distribution for all inputs. This happens because:

1. The teacher has no gradients - it can drift toward any constant output
2. The student then learns to match this constant, achieving low loss without learning useful features

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

**EMA update benefits**:
- Smooth updates prevent oscillation
- Batch-to-batch variations are dampened
- The center represents a stable "average behavior" rather than reacting to individual batches

### View Asymmetry

The key asymmetry in DINO:

- **Teacher**: Only sees **global views** (2 large 224×224 crops covering 40-100% of image)
- **Student**: Sees **all views** (2 global + 6 local 96×96 crops covering 5-40% of image)

```
Teacher sees:        Student sees:
┌─────────────┐     ┌─────────────┐
│ ┌─────────┐ │     │ ┌─────────┐ │
│ │ Global  │ │     │ │ Global  │ │  + 6 local crops:
│ │  View   │ │     │ │  View   │ │  ┌───┐ ┌───┐ ┌───┐
│ └─────────┘ │     │ └─────────┘ │  │   │ │   │ │   │
└─────────────┘     └─────────────┘  └───┘ └───┘ └───┘
    (×2)                (×2)            (×6)
```

**Why this asymmetry matters**:

1. **Local-to-global correspondence**: The student must predict teacher's global view output from local crops. This forces the student to learn that small patches belong to the same semantic object as the full image.

2. **Multi-scale reasoning**: The student learns features that are consistent across different scales (5% to 100% of the image).

3. **Information bottleneck**: The teacher has more context (sees large regions), creating an information asymmetry. The student must extract more meaning from less context.

4. **Cross-view loss computation**: Each teacher view (2 total) is compared against each student view (8 total), excluding same-index pairs. This creates 2×8 - 2 = 14 loss terms per sample.

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
