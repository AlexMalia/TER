# Training API Reference

API documentation for the training module.

---

## DinoTrainer

::: dino.training.DinoTrainer
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - train
        - train_epoch
        - train_step

---

## Optimizer Helpers

### create_optimizer

::: dino.training.create_optimizer
    options:
      show_root_heading: true
      show_source: true

### create_scheduler

::: dino.training.create_scheduler
    options:
      show_root_heading: true
      show_source: true

---

## Usage Examples

### Complete Training Setup

```python
from dino.config import DinoConfig
from dino.models import DinoModel
from dino.loss import DinoLoss
from dino.training import DinoTrainer, create_optimizer, create_scheduler
from dino.data import create_dataloaders

# Load configuration
config = DinoConfig.from_yaml('configs/default.yaml')

# Create data loaders
train_loader, val_loader, _ = create_dataloaders(config)

# Create models
student = DinoModel.from_config(config)
teacher = DinoModel.from_config(config)
teacher.load_state_dict(student.state_dict())

# Disable gradients for teacher
for param in teacher.parameters():
    param.requires_grad = False

# Create loss
loss_fn = DinoLoss.from_config(
    config.loss,
    config.augmentation,
    out_dim=student.output_dim
)

# Create optimizer and scheduler
optimizer = create_optimizer(student.parameters(), config.optimizer)

# Calculate steps accounting for gradient accumulation
accumulation_steps = config.training.gradient_accumulation_steps
updates_per_epoch = len(train_loader) // accumulation_steps
total_steps = config.training.num_epochs * updates_per_epoch
warmup_steps = config.scheduler.warmup_epochs * updates_per_epoch

scheduler = create_scheduler(
    optimizer,
    config.scheduler,
    config.optimizer,
    total_steps,
    warmup_steps
)

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

### Creating Optimizer

```python
from dino.training import create_optimizer

optimizer = create_optimizer(
    model.parameters(),
    config.optimizer
)

# config.optimizer contains:
# - optimizer: 'adamw'
# - lr: 0.001
# - weight_decay: 0.04
# - betas: [0.9, 0.999]
```

### Creating Scheduler

```python
from dino.training import create_scheduler

scheduler = create_scheduler(
    optimizer,
    config.scheduler,
    config.optimizer,
    total_steps=num_epochs * steps_per_epoch,
    warmup_steps=warmup_epochs * steps_per_epoch
)

# config.scheduler contains:
# - scheduler: 'cosine_warmup'
# - warmup_epochs: 10
# - min_lr: 1e-6
# - warmup_start_lr: 0.0
```

### Training Configuration

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
  lr: 0.001
  weight_decay: 0.04
  betas: [0.9, 0.999]

scheduler:
  scheduler: cosine_warmup
  warmup_epochs: 10
  min_lr: 1.0e-6
  warmup_start_lr: 0.0
```
