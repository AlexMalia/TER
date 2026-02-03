# DINO: Self-Supervised Vision Learning

Learn powerful visual representations **without labels** using DINO (Self-DIstillation with NO labels). This is a clean, production-ready PyTorch implementation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexMalia/TER.git
cd DinoImpl

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Train Your First Model

```bash
# Train on ImageNette (default)
python scripts/train.py

# Train on ImageNet100
python scripts/train.py --config configs/imagenet100.yaml

# Train with custom settings
python scripts/train.py --epochs 100 --batch-size 64 --lr 0.001
```

That's it! Your model will train and save checkpoints to `./checkpoints/`.

---

## Features

- **Easy to use**: Train with a single command
- **Production ready**: Clean code structure, proper logging, checkpointing
- **Configurable**: YAML configs for all hyperparameters
- **Extensible**: Easy to add new datasets or backbones
- **Multiple datasets**: ImageNette and ImageNet100 included
- **Training history**: Track and visualize loss, learning rate, and momentum

---

## Documentation

Full documentation is available at: **[Documentation Site](https://alexmalia.github.io/TER/DinoImpl/)**

- [Installation Guide](https://alexmalia.github.io/TER/DinoImpl/getting-started/installation/)
- [Quick Start](https://alexmalia.github.io/TER/DinoImpl/getting-started/quickstart/)
- [Configuration Reference](https://alexmalia.github.io/TER/DinoImpl/guides/configuration/)
- [API Reference](https://alexmalia.github.io/TER/DinoImpl/api/models/)
- [Troubleshooting](https://alexmalia.github.io/TER/DinoImpl/troubleshooting/)

### Build Documentation Locally

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve locally with hot-reload
mkdocs serve

# Build static site
mkdocs build
```

---

## Project Structure

```
DinoImpl/
├── src/dino/              # Main package
│   ├── config/           # Configuration management
│   ├── data/             # Data loading & augmentation
│   ├── models/           # Neural network models
│   ├── loss/             # Loss functions
│   ├── training/         # Training logic
│   └── utils/            # Utilities
├── configs/              # YAML configuration files
├── scripts/              # Executable scripts
├── docs/                 # Documentation (MkDocs)
└── checkpoints/          # Saved models (auto-created)
```

