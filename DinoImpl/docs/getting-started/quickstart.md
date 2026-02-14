# Quick Start

Get up and running with DINO.

---

## Train Your First Model

### Basic Training

```bash
# Train on default config
uv run scripts/train.py
```

That's it! Your model will train and save checkpoints to `./output/checkpoints/`.

---

## Training Options

### With Configuration File

```bash
uv run scripts/train.py --config configs/default.yaml
```

### Override Parameters

Parameters will override the settings set in your YAML configuration file.

```bash
uv run scripts/train.py \
    --config configs/imagenet100.yaml \
    --batch-size 64 \
    --epochs 200 \
    --lr 0.0005
```

Check the [Configuration](../guides/configuration.md) guide for more details.

### Resume from Checkpoint

```bash
uv run scripts/train.py --resume <checkpoint-path>
```

---

## Using DINO in Python

### Simple Usage with Factory Methods

```python
from dino.config import DinoConfig
from dino.models import DinoModel
from dino.loss import DinoLoss
from dino.training import DinoTrainer, create_optimizer, create_scheduler
from dino.data import create_dataloaders

# Load configuration
config = DinoConfig.from_yaml_and_args('configs/default.yaml', args)

# Create components using factory methods
train_loader, val_loader, _ = create_dataloaders(config.data_config, config.augmentation_config)
student = DinoModel.from_config(config.model_config)
teacher = DinoModel.from_config(config.model_config)

teacher.load_state_dict(student.state_dict())

# Create loss and optimizer
dino_loss = DinoLoss.from_config(
    config.loss_config,
    config.augmentation_config,
    out_dim=student.output_dim
)
optimizer = create_optimizer(student.parameters(), config.optimizer_config)

# Create scheduler (accounts for gradient accumulation)
accumulation_steps = config.training.gradient_accumulation_steps
updates_per_epoch = len(train_loader) // accumulation_steps
total_steps = config.training.num_epochs * updates_per_epoch
warmup_steps = config.scheduler.warmup_epochs * updates_per_epoch
scheduler = create_scheduler(optimizer, config.scheduler_config, config.optimizer_config, total_steps, warmup_steps)

# Train
trainer = DinoTrainer(
    config=config,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    train_loader=train_loader
)
trainer.train()
```

## Monitoring Training

### Console Output

```
[INFO] Epoch 1/100
Epoch 1/100: 100%|████████| 208/208 [02:15<00:00]  loss: 8.2341, momentum: 0.9960
[INFO] Train Epoch 1 - loss: 8.2145, momentum: 0.9960
```

### Weights & Biases

Enable W&B for rich experiment tracking:

```bash
# First, login to W&B
wandb login

# Train with W&B enabled
python scripts/train.py --config configs/default.yaml
```

in your config you need atleast:

```yaml
logging:
  use_wandb: true
  wandb_project: dino-training
```

---

## Training History

Track and visualize training metrics:

```python
from dino.utils import History

# Load history from a training run
history = History.load('training_history.json')

# Plot metrics
history.plot_loss(level='epoch')
history.plot_learning_rate(level='iteration')
history.plot_all(level='epoch', save_path='training_plots.png')

# Export to DataFrame (requires pandas)
df = history.to_dataframe(level='epoch')
print(df.head())
```

---

## What's Next?

- [CLI Reference](cli-reference.md) - All command-line options
- [Configuration](../guides/configuration.md) - Customize your training
- [Models](../guides/models.md) - Available backbones and architectures
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
