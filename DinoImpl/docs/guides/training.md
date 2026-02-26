# Training

This guide explains the DINO training pipeline, including the training loop, EMA updates, scheduler, and optimizers.

---

## Overview

The DINO training pipeline:

1. **Forward pass**: Student sees all views, teacher sees global views only
2. **Loss computation**: Cross-entropy between student and teacher predictions
3. **Backward pass**: Gradients only through student
4. **EMA update**: Teacher weights updated from student via exponential moving average


---

## Initialization Sequence

The complete initialization flow:

1. **Configuration**: Load YAML → Override with CLI args
2. **Logging**: Setup file and console handlers
3. **Data**: Create dataloaders with `create_dataloaders(config)`
4. **Models**: Create with `DinoModel.from_config(config)`
5. **Loss**: Create with `DinoLoss.from_config(config.loss, config.augmentation, out_dim)`
6. **Optimizer**: Create with `create_optimizer(params, config.optimizer)`
7. **Scheduler**: Create with `create_scheduler(...)`
8. **Trainer**: Assemble all components
9. **Resume** (optional): Load checkpoint if resuming
10. **Setup W&B** (optional): Initialize W&B run and config
11. **Train**: Run training loop

---

## Training Loop

### High-Level Flow

```python
for epoch in range(num_epochs):
    for batch_idx, (views, _) in enumerate(train_loader):
        # 1. Separate views
        global_views, all_views = get_global_local_views(view_set)

        # 2. Teacher forward (no gradients)
        with torch.no_grad():

          teacher_output = torch.cat(
            [teacher(v) for v in global_views], 
            dim=0
          ) # [num_global_views * batch_size, output_dim]

        # 3. Student forward (with gradients)
        student_output = torch.cat(
          [student(v) for v in all_views], 
          dim=0
        ) # [num_all_views * batch_size, output_dim]

        # 4. Compute loss
        loss = dino_loss(student_output, teacher_output)

        # 5. Backward pass
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(student.parameters(), max_norm)  # Optional
        optimizer.step()

        # 6. Learning rate scheduler step
        scheduler.step()

        # 7. EMA update for teacher
        update_teacher_EMA(student, teacher, momentum)
```

### Using DinoTrainer

```python
from dino.training import DinoTrainer, create_optimizer, create_scheduler
from dino.models import DinoModel
from dino.loss import DinoLoss
from dino.data import create_dataloaders

# Setup
config = DinoConfig.from_yaml('config.yaml')
train_loader, val_loader, _ = create_dataloaders(config)

student = DinoModel.from_config(config)
teacher = DinoModel.from_config(config)
teacher.load_state_dict(student.state_dict())

loss_fn = DinoLoss.from_config(config.loss, config.augmentation, student.output_dim)
optimizer = create_optimizer(student.parameters(), config.optimizer)
scheduler = create_scheduler(optimizer, config.scheduler, config.optimizer, total_steps, warmup_steps)

# Create trainer
trainer = DinoTrainer(
    config=config,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    train_loader=train_loader
)

# Train
trainer.train()
```

---

## EMA (Exponential Moving Average)

### How EMA Works

The teacher is updated as a moving average of the student:

```python
teacher = momentum * teacher + (1 - momentum) * student
```

```python
@torch.no_grad()
def update_teacher_EMA(student, teacher, momentum=0.996):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data = momentum * t_param.data + (1 - momentum) * s_param.data
```

Current weight matrix is equal to 99.6 of the previous teacher weights + 0.4 of the current student weights.

Do not confuse teacher momentum and center momentum. The first updates the teacher's network weights, while the second updates the centering vector to prevent trivial solution and collapse.

### Momentum Scheduling


The momentum typically increases during training:

- **Early training**: Lower momentum (0.996) - teacher learns faster
- **Late training**: Higher momentum (0.999) - teacher becomes stable

```python
from dino.utils import get_momentum_schedule

schedule = get_momentum_schedule(
    base=0.996,
    final=1.0,
    num_epochs=100,
    niter_per_epoch=len(train_loader)
)

# During training
momentum = schedule[current_iteration]
update_teacher_EMA(student, teacher, momentum)
```

For all schedules, we use a cosine schedule. Source code for the schedule can be found in `dino/utils/schedule.py`.

### Configuration

```yaml
training:
  teacher_momentum: 0.996
  teacher_momentum_final: 1.0
  teacher_momentum_schedule: true
```

Dino original implementation uses a cosine schedule that starts at 0.996 and ends at 1.0.

---

## Optimizer

### Supported Optimizers

Currently supports AdamW, Adam and SGD:

```python
from dino.training import create_optimizer

optimizer = create_optimizer(
    model.parameters(),
    config.optimizer
)
```

### Weight decay

We use a fixed weight decay, to follow the original DINO implementation we must use a scheduled weight decay that starts at 0.04 and ends at 0.4.

### Configuration

```yaml
optimizer:
  optimizer: adamw
  lr: 0.0005
  weight_decay: 0.04
  betas: [0.9, 0.999]
```

---

## Learning Rate Scheduler

### Cosine with Warmup

Supports cosine_warmup, cosine, linear and constant

```python
from dino.training import create_scheduler

scheduler = create_scheduler(
    optimizer,
    config.scheduler,
    config.optimizer,
    total_steps=num_epochs * len(train_loader),
    warmup_steps=warmup_epochs * len(train_loader)
)
```

### Configuration

```yaml
scheduler:
  scheduler: cosine_warmup
  warmup_epochs: 10
  min_lr: 1e-6
  warmup_start_lr: 0.0
```

### How It Works

1. **Warmup phase**: Linear increase from `warmup_start_lr` to `optimizer.lr`
2. **Cosine phase**: Cosine decay from `optimizer.lr` to `min_lr`

---

Dino original implementation uses adamw with a cosine LR schedule with linear warmup for the first 10 epochs. The default learning rate is 0.0005 for batch size 256 and scale linearly with batch size, the min learning rate (target learning rate at the end of cosine decay) is 1e-6.

## Gradient Clipping

Prevents exploding gradients during training:

```python
if config.training.gradient_clip:
    torch.nn.utils.clip_grad_norm_(
        student.parameters(),
        config.training.gradient_clip
    )
```

### Configuration

```yaml
training:
  gradient_clip: 3.0
```

Dino original implementation uses a max gradient norm of 3.0.

---

## Training Configuration

Values are exemples. Adjust based on your dataset, model size and GPU resources.

```yaml
training:
  num_epochs: 100
  teacher_momentum: 0.996
  teacher_momentum_final: 1.0
  teacher_momentum_schedule: true
  gradient_clip: 3.0
  gradient_accumulation_steps: 1

optimizer:
  optimizer: adamw
  lr: 0.0005
  weight_decay: 0.04
  betas: [0.9, 0.999]
  eps: 1.0e-08

scheduler:
  scheduler: cosine_warmup
  warmup_epochs: 10
  min_lr: 1e-6
```

---

## Training History


The `History` class has 2 purposes : 
1. **Logging**: Track metrics like loss, learning rate, momentum at each iteration and epoch
2. **Visualization**: Provides tools to print and plot training curves for analysis

The `History` class tracks metrics during training:

```python
from dino.utils import History

history = History(metadata={<Info about the training run, e.g. config, device, etc.>})

1. Logging metrics during training
# Record during training
history.record_iteration(iteration, {
    'loss': loss.item(),
    'learning_rate': optimizer.param_groups[0]['lr'],
    'momentum': current_momentum
})

history.record_epoch(epoch, {
    'loss': epoch_loss,
    'learning_rate': current_lr,
    'momentum': current_momentum
})

history.save('training_history.json')


2. Provide visualization tools
# Save/Load
history = History.load('training_history.json')

# Visualization
history.plot_all(level='epoch', save_path='plots.png')
```

  Note : an improvment of this would be to split the history into 2 classes, each with a single responsibility, one for logging and one for visualization. This would make the code cleaner and easier to maintain. 

---

## Gradient Accumulation

When training with limited GPU memory, gradient accumulation allows you to simulate larger batch sizes:

```yaml
training:
  gradient_accumulation_steps: 4  # Effective batch = batch_size × 4
```

**How it works**:
1. Forward pass and loss computation for each mini-batch
2. Divide loss by `accumulation_steps` to normalize gradients
3. Backward pass (gradients accumulate in-place across steps)
4. After N steps: clip gradients, run optimizer, update LR scheduler, update teacher EMA, then zero gradients

```python
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / gradient_accumulation_steps
    loss.backward()

    if (i + 1) % gradient_accumulation_steps == 0:
        clip_grad_norm_(student.parameters(), max_norm)  # clipping on accumulated gradients
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        update_teacher_EMA(student, teacher, momentum)
        current_iteration += 1
```

**Use case**: With `batch_size=8` and `gradient_accumulation_steps=4`, you get the same gradient statistics as `batch_size=32` but using 4× less GPU memory per step.

---

## Weights & Biases Integration

Track experiments with [Weights & Biases](https://wandb.ai):

### Configuration

```yaml
logging:
  use_wandb: true
  wandb_project: dino-training
  wandb_entity: your-username    # Optional: W&B username or team
  wandb_run_name: experiment-1   # Optional: custom run name
```

### What Gets Logged

- **Iteration metrics**: Loss, learning rate, momentum (per step)
- **Epoch metrics**: Average loss, learning rate, momentum (per epoch)
- **Configuration**: Full training config saved as W&B config
- **Tags**: Backbone type, dataset name

### Resuming with W&B

When resuming from a checkpoint, the W&B run ID is automatically restored:

```python
# Checkpoint includes wandb_run_id
# On resume, training continues logging to the same W&B run
trainer.resume_from_checkpoint('checkpoint.pth')
```

### Manual Setup

```python
import wandb

wandb.init(
    project="dino-training",
    config=config.to_dict(),
    tags=[config.model.backbone, config.data.dataset]
)

# During training, metrics are logged automatically by DinoTrainer
```

---

## Kaggle Training

Train DINO models on Kaggle with GPU acceleration.

### Available Configurations

| Config | Dataset | Backbone | Notes |
|--------|---------|----------|-------|
| `kaggle-imagenette.yaml` | ImageNette | ViT-S/16 | Quick experiments |
| `kaggle-imagenet100.yaml` | ImageNet100 | ViT-S/16 | Full training |

### Using Kaggle Configurations

```bash
./kaggle/kaggle_manager.sh push
```

### Key Differences for Kaggle

1. **Data paths**: Use `/kaggle/input/` for datasets
2. **Output paths**: Use `/kaggle/working/` for checkpoints and logs
3. **Workers**: Reduced to 2 (Kaggle container limits)
4. **Gradient accumulation**: Enabled to simulate larger batches

### Example Kaggle Config

```yaml
data:
  dataset: imagenet100
  data_path: /kaggle/input/imagenet100
  num_workers: 2

training:
  gradient_accumulation_steps: 2

checkpoint:
  checkpoint_dir: /kaggle/working/checkpoints

logging:
  use_wandb: true
  wandb_project: dino-training
```

---

## See Also

- [Loss Function](loss-function.md) - DINO loss computation
- [Checkpointing](checkpointing.md) - Saving and resuming training
- [Configuration](configuration.md) - Full configuration reference
- [Performance](../advanced/performance.md) - Optimization tips
