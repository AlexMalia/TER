# Performance Optimization

Guide to optimizing training performance, memory usage, and speed.

---

## Memory Optimization

### Multi-Crop Memory Impact

DINO's multi-crop strategy uses 8 views per image, significantly increasing memory usage:

```
Standard training: batch_size × 1 view
DINO training:     batch_size × 8 views = 8× memory
```

### Reducing Memory Usage

#### 1. Reduce Batch Size

```yaml
data:
  batch_size: 16  # Instead of 64
```

#### 2. Reduce Local Crops

```yaml
augmentation:
  num_local_views: 4  # Instead of 6
```

#### 3. Use Smaller Backbone

```yaml
model:
  backbone: resnet18  # Instead of resnet50
```

#### 4. Gradient Accumulation

Simulate larger batch sizes without extra memory:

```python
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Training Speed

### DataLoader Optimization

```yaml
data:
  num_workers: 8        # Match CPU cores
  pin_memory: true      # Faster GPU transfer
  persistent_workers: true  # Keep workers alive
```

### Bottleneck Analysis

Common bottlenecks and solutions:

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| Data loading | GPU utilization < 90% | Increase `num_workers` |
| Augmentation | CPU at 100% | Cache augmented data |
| Forward pass | Normal | Use mixed precision |
| Backward pass | Normal | Use gradient checkpointing |

---

## Mixed Precision Training

Use FP16 to reduce memory and increase speed:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast():
        student_output = student(views)
        teacher_output = teacher(global_views)
        loss = loss_fn(student_output, teacher_output)

    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # EMA update (in FP32)
    update_teacher_EMA(student, teacher, momentum)
```

### Benefits

- **Memory**: ~50% reduction
- **Speed**: 2-3× faster on modern GPUs
- **Accuracy**: Typically no degradation

---

## Gradient Checkpointing

Trade compute for memory by recomputing activations:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        # Recompute activations during backward
        return checkpoint(self.backbone, x)
```

### When to Use

- Large models (ResNet50+, ViT)
- GPU memory is the limiting factor
- Willing to trade ~20% speed for ~30% memory savings

---

## Distributed Training

### Data Parallel (Single Node, Multi-GPU)

```python
import torch.nn as nn

# Wrap student only (teacher doesn't need gradients)
student = nn.DataParallel(student)
```

### Distributed Data Parallel (Multi-Node)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# Wrap model
student = DDP(student, device_ids=[local_rank])

# Synchronize center in loss
if dist.is_initialized():
    dist.all_reduce(batch_center)
    batch_center /= dist.get_world_size()
```

### Launch Command

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 scripts/train.py

# Multi-node
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_endpoint=master:29500 \
    scripts/train.py
```

---

## Profiling

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler('./logs/profile'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        train_step(batch)
        prof.step()

# View in TensorBoard
# tensorboard --logdir logs/profile
```

### NVIDIA Tools

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Detailed profiling
nsys profile python scripts/train.py
```

---

## Hardware Recommendations

### Minimum (Small experiments)

- GPU: 8GB VRAM (GTX 1080, RTX 2080)
- RAM: 16GB
- Storage: SSD recommended

### Recommended (Full training)

- GPU: 16GB+ VRAM (RTX 3090, A100)
- RAM: 32GB+
- Storage: NVMe SSD

### Configurations by GPU

| GPU | Batch Size | Backbone | Local Crops |
|-----|------------|----------|-------------|
| 8GB | 16 | ResNet18 | 4 |
| 12GB | 32 | ResNet18 | 6 |
| 16GB | 32 | ResNet50 | 6 |
| 24GB+ | 64 | ResNet50/ViT | 6-8 |

---

## Quick Optimization Checklist

- [ ] Set `num_workers` to CPU core count
- [ ] Enable `pin_memory: true`
- [ ] Use appropriate batch size for GPU
- [ ] Consider reducing local crops (6 → 4)
- [ ] Enable mixed precision for modern GPUs
- [ ] Use gradient accumulation if batch size limited
- [ ] Profile to identify bottlenecks

---

## See Also

- [Training](../guides/training.md) - Training configuration
- [Configuration](../guides/configuration.md) - Full config reference
- [Troubleshooting](../troubleshooting.md) - Common issues
