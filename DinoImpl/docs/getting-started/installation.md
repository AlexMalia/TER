# Installation

This guide covers the installation of DINO and its dependencies.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

---

## Quick Install

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast, modern Python package manager.

```bash
# Clone the repository
git clone <your-repo-url>
cd DinoImpl

# Install with uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone <your-repo-url>
cd DinoImpl

# Install in development mode
pip install -e .
```

---

## Optional Dependencies

### Documentation

To build the documentation locally:

```bash
uv sync --extras docs
```

### Development Tools

For linting, testing, and code formatting:

```bash
uv sync --extras dev
```

### Weights & Biases

W&B is included in the base installation. To use it:

```bash
# Login to W&B (required once)
wandb login

# Enable in config
logging:
  use_wandb: true
  wandb_project: dino-training
```

Or install separately:

```bash
uv sync --extras wandb
```

### Jupyter Notebooks

For running notebooks:

```bash
uv sync --extras notebooks
```


### Kaggle

To train on Kaggle GPUs

```bash
uv sync --extras kaggle
```

---

## Verifying Installation

After installation, verify everything works:

```bash
# Check the module is importable
uv run python -c 'from dino import DinoConfig, DinoModel; print("Installation successful!")'

# Check available commands
uv run python scripts/train.py --help
```

---

## Dataset Setup

### ImageNette (Default)

ImageNette downloads automatically when you first run training:

```bash
uv run python scripts/train.py
# Dataset will be downloaded to ./data/imagenette2/
```

### ImageNet100

ImageNet100 requires manual download from Kaggle:

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (requires account)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d ambityga/imagenet100

# Extract to data directory
unzip imagenet100.zip -d ./data/imagenet100
```

See [Data Pipeline](../guides/data-pipeline.md) for more dataset options.

---

## Next Steps

- [Quick Start](quickstart.md) - Train your first model
- [CLI Reference](cli-reference.md) - Command-line options
- [Configuration](../guides/configuration.md) - Full configuration guide
