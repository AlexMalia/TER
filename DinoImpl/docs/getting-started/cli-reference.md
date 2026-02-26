# CLI Reference

Complete reference for command-line options. Those options can be used to override config values when running training scripts.

---

## Training Script

```bash
uv run scripts/train.py [OPTIONS]
```

### Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | `configs/default.yaml` | Path to configuration file |
| `--resume` | path | None | Resume from checkpoint |

### Data Options

| Option | Type | Description |
|--------|------|-------------|
| `--dataset` | str | Dataset name (imagenette, imagenet100) |
| `--data-path` | path | Path to data directory |
| `--batch-size` | int | Batch size for training |
| `--num-workers` | int | Number of data loading workers |

### Training Options

| Option | Type | Description |
|--------|------|-------------|
| `--epochs` | int | Number of training epochs |
| `--lr` | float | Learning rate |
| `--seed` | int | Random seed for reproducibility |
| `--device` | str | Device to use (cuda, cpu) |

### Model Options

| Option | Type | Description |
|--------|------|-------------|
| `--backbone` | str | Backbone architecture |
| `--output-dim` | int | Projection output dimension |

### Logging Options

| Option | Type | Description |
|--------|------|-------------|
| `--checkpoint-dir` | path | Checkpoint directory |
| `--log-dir` | path | Log directory |
| `--log-verbosity` | str | Log level (debug, info, warning, error) |

---

## Examples

### Basic Training

```bash
# Train with defaults
uv run scripts/train.py

# Train with specific config
uv run scripts/train.py --config configs/imagenet100.yaml
```

### Override Parameters

```bash
# Smaller batch size for limited GPU memory (Be careful, dino needs large batch sizes for good performance)
uv run scripts/train.py --batch-size 16

# Longer training with lower learning rate
uv run scripts/train.py --epochs 200 --lr 0.0005

# Use ViT backbone
uv run scripts/train.py --backbone dino_vits16
```

### Resume Training

```bash
# Resume from latest checkpoint
uv run scripts/train.py --resume checkpoints/checkpoint_latest.pth

# Resume with modified parameters
uv run scripts/train.py --resume checkpoints/checkpoint_epoch_50.pth --lr 0.0001
```

### Multiple Overrides

```bash
uv run scripts/train.py \
    --config configs/imagenet100.yaml \
    --batch-size 64 \
    --epochs 200 \
    --lr 0.0005 \
    --num-workers 8
```

---

## Priority Order

Command-line arguments override configuration file values:

1. **CLI arguments** (highest priority)
2. **YAML configuration file**
3. **Default config** (if no --config option) (lowest priority)

Example:
```bash
# config.yaml has batch_size: 32
# CLI overrides to 64
uv run scripts/train.py --config config.yaml --ba
tch-size 64
# Result: batch_size = 64
```

## See Also

- [Configuration](../guides/configuration.md) - Full YAML configuration reference
- [Training](../guides/training.md) - Training pipeline details
- [Checkpointing](../guides/checkpointing.md) - Checkpoint management
