# Quick Start

Get up and running with DINO in 5 minutes.

---

## Train Your First Model

### Basic Training

```bash
# Train on ImageNette (default dataset)
python scripts/train.py
```

That's it! Your model will train and save checkpoints to `./checkpoints/`.

### Training with Different Datasets

```bash
# Train on ImageNet100
python scripts/train.py --config configs/imagenet100.yaml
```

### Custom Settings

```bash
# Override specific parameters
python scripts/train.py --epochs 100 --batch-size 64 --lr 0.001
```

---

## Training Options

### With Configuration File

```bash
python scripts/train.py --config configs/default.yaml
```

### Override Parameters

```bash
python scripts/train.py \
    --config configs/imagenet100.yaml \
    --batch-size 64 \
    --epochs 200 \
    --lr 0.0005
```

### Resume from Checkpoint

```bash
python scripts/train.py --resume checkpoints/checkpoint_latest.pth
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
config = DinoConfig.from_yaml('configs/default.yaml')

# Create components using factory methods
train_loader, val_loader, _ = create_dataloaders(config)
student = DinoModel.from_config(config)
teacher = DinoModel.from_config(config)
teacher.load_state_dict(student.state_dict())

# Create loss and optimizer
loss_fn = DinoLoss.from_config(config.loss, config.augmentation, out_dim=student.output_dim)
optimizer = create_optimizer(student.parameters(), config.optimizer)

# Train
trainer = DinoTrainer(
    config=config,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    scheduler=None,
    loss_fn=loss_fn,
    train_loader=train_loader
)
trainer.train()
```

### Manual Component Creation

For more control over the components:

```python
from dino.models import get_backbone, DinoProjectionHead, DinoModel

# Create components manually
backbone = get_backbone('resnet18')
projection = DinoProjectionHead(input_dim=512, output_dim=2048)
student = DinoModel(backbone, projection)
```

---

## Monitoring Training

### Console Output

```
[INFO] Epoch 1/100
Epoch 1/100: 100%|████████| 208/208 [02:15<00:00]  loss: 8.2341, momentum: 0.9960
[INFO] Train Epoch 1 - loss: 8.2145, momentum: 0.9960
```

### TensorBoard

```bash
tensorboard --logdir logs/
```

### Check Checkpoints

```bash
ls -lh checkpoints/
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
