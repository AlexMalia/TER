# DINO: Self-Supervised Vision Learning

Learn powerful visual representations **without labels** using DINO (Self-DIstillation with NO labels). This is a clean, production-ready PyTorch implementation.

---

## What is DINO?

DINO teaches a neural network to recognize images **without any labels**. It works by:

1. Taking one image and creating multiple different views (crops, color changes, etc.)
2. A "student" network tries to predict what a "teacher" network sees
3. The teacher slowly learns from the student's improvements

**Result**: The network learns powerful visual features that work for many downstream tasks (classification, detection, segmentation).

---

## Features

- **Easy to use**: Train with a single command
- **Production ready**: Clean code structure, proper logging, checkpointing
- **Configurable**: YAML configs for all hyperparameters
- **Extensible**: Easy to add new datasets or backbones
- **Bug-free**: Fixed the negative loss bug from the original notebook
- **Multiple datasets**: ImageNette and ImageNet100 included
- **Training history**: Track and visualize loss, learning rate, and momentum
- **W&B integration**: Experiment tracking with Weights & Biases
- **Kaggle support**: Ready-to-use configurations for Kaggle training
- **Gradient accumulation**: Train with larger effective batch sizes on limited GPU memory

---

## Quick Links

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Getting Started**

    ---

    Install DINO and train your first model in minutes

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } **Guides**

    ---

    Learn about data pipelines, models, training, and more

    [:octicons-arrow-right-24: Data Pipeline](guides/data-pipeline.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Complete API documentation generated from source code

    [:octicons-arrow-right-24: API Reference](api/models.md)

-   :material-wrench:{ .lg .middle } **Troubleshooting**

    ---

    Solutions to common problems

    [:octicons-arrow-right-24: Troubleshooting](troubleshooting.md)

</div>

---

## Project Structure

```
DinoImpl/
├── src/dino/              # Main package
│   ├── config/           # Configuration management
│   ├── data/             # Data loading & augmentation
│   ├── models/           # Neural network models
│   │   └── backbone/     # Backbone architectures (ResNet, ViT)
│   ├── loss/             # Loss functions
│   ├── training/         # Training logic
│   ├── evaluation/       # Evaluation utilities
│   └── utils/            # Utilities (checkpointing, logging, history, etc.)
│
├── configs/              # YAML configuration files
├── scripts/              # Executable scripts
├── checkpoints/          # Saved model checkpoints (auto-created)
├── logs/                 # Training logs (auto-created)
└── data/                 # Datasets (auto-downloaded)
```

---

## How It Works (Simple Explanation)

1. **Multi-Crop Augmentation**: Takes one image, creates 8 different views (2 large, 6 small)
2. **Student Network**: Looks at all 8 views and makes predictions
3. **Teacher Network**: Only looks at the 2 large views
4. **Loss**: Student tries to match teacher's predictions
5. **EMA Update**: Teacher slowly copies student's weights
6. **Centering**: Prevents the model from always predicting the same thing

**Why this works:** The student has to learn robust features that work across different views, scales, and augmentations.

---

## Learn More

- **Original Paper**: [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- **Official Implementation**: [facebookresearch/dino](https://github.com/facebookresearch/dino)
