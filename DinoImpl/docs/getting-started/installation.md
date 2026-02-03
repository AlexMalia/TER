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
pip install -e ".[docs]"
# or
uv pip install -e ".[docs]"
```

### Development Tools

For linting, testing, and code formatting:

```bash
pip install -e ".[dev]"
# or
uv pip install -e ".[dev]"
```

### Weights & Biases

For experiment tracking with W&B:

```bash
pip install -e ".[wandb]"
```

### Jupyter Notebooks

For running notebooks:

```bash
pip install -e ".[notebooks]"
```

---

## Verifying Installation

After installation, verify everything works:

```bash
# Check the module is importable
python -c "from dino import DinoConfig, DinoModel; print('Installation successful!')"

# Check available commands
python scripts/train.py --help
```

---

## GPU Setup

DINO benefits significantly from GPU acceleration. Ensure your PyTorch installation includes CUDA support:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

If CUDA is not available, reinstall PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Dataset Setup

### ImageNette (Default)

ImageNette downloads automatically when you first run training:

```bash
python scripts/train.py
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
