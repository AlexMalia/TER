# CLI Reference

Complete reference for command-line options.

---

## Training Script

```bash
python scripts/train.py [OPTIONS]
```

### Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | `configs/default.yaml` | Path to configuration file |
| `--resume` | path | None | Resume from checkpoint |

### Data Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | str | imagenette | Dataset name (imagenette, imagenet100) |
| `--data-path` | path | ./data | Path to data directory |
| `--batch-size` | int | 32 | Batch size for training |
| `--num-workers` | int | 4 | Number of data loading workers |

### Training Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--epochs` | int | 100 | Number of training epochs |
| `--lr` | float | 0.001 | Learning rate |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--device` | str | cuda | Device to use (cuda, cpu) |

### Model Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--backbone` | str | resnet18 | Backbone architecture |
| `--output-dim` | int | 2048 | Projection output dimension |

### Logging Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint-dir` | path | ./checkpoints | Checkpoint directory |
| `--log-dir` | path | ./logs | Log directory |
| `--log-verbosity` | str | info | Log level (debug, info, warning, error) |

---

## Examples

### Basic Training

```bash
# Train with defaults
python scripts/train.py

# Train with specific config
python scripts/train.py --config configs/imagenet100.yaml
```

### Override Parameters

```bash
# Smaller batch size for limited GPU memory
python scripts/train.py --batch-size 16

# Longer training with lower learning rate
python scripts/train.py --epochs 200 --lr 0.0005

# Use ViT backbone
python scripts/train.py --backbone dino_vits16
```

### Resume Training

```bash
# Resume from latest checkpoint
python scripts/train.py --resume checkpoints/checkpoint_latest.pth

# Resume with modified parameters
python scripts/train.py --resume checkpoints/checkpoint_epoch_50.pth --lr 0.0001
```

### Multiple Overrides

```bash
python scripts/train.py \
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
3. **Default values** (lowest priority)

Example:
```bash
# config.yaml has batch_size: 32
# CLI overrides to 64
python scripts/train.py --config config.yaml --batch-size 64
# Result: batch_size = 64
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | GPUs to use (e.g., "0,1") |
| `TORCH_HOME` | PyTorch cache directory |

Example:
```bash
# Use only GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/train.py

# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py
```

---

## See Also

- [Configuration](../guides/configuration.md) - Full YAML configuration reference
- [Training](../guides/training.md) - Training pipeline details
- [Checkpointing](../guides/checkpointing.md) - Checkpoint management
