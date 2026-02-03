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
git clone <your-repo-url>
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

Full documentation is available at: **[Documentation Site](https://your-username.github.io/DinoImpl/)**

- [Installation Guide](https://your-username.github.io/DinoImpl/getting-started/installation/)
- [Quick Start](https://your-username.github.io/DinoImpl/getting-started/quickstart/)
- [Configuration Reference](https://your-username.github.io/DinoImpl/guides/configuration/)
- [API Reference](https://your-username.github.io/DinoImpl/api/models/)
- [Troubleshooting](https://your-username.github.io/DinoImpl/troubleshooting/)

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

---

## Citation

If you use this code in your research, please cite the original DINO paper:

```bibtex
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={ICCV},
  year={2021}
}
```

---

## License

MIT License - see LICENSE file for details.
