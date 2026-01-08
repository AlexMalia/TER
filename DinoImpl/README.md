# DINO: Self-Supervised Vision Learning

Learn powerful visual representations **without labels** using DINO (Self-DIstillation with NO labels). This is a clean, production-ready PyTorch implementation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 What is DINO?

DINO teaches a neural network to recognize images **without any labels**. It works by:
1. Taking one image and creating multiple different views (crops, color changes, etc.)
2. A "student" network tries to predict what a "teacher" network sees
3. The teacher slowly learns from the student's improvements

**Result**: The network learns powerful visual features that work for many downstream tasks (classification, detection, segmentation).

---

## ✨ Features

- 🚀 **Easy to use**: Train with a single command
- 📦 **Production ready**: Clean code structure, proper logging, checkpointing
- ⚙️ **Configurable**: YAML configs for all hyperparameters
- 🔧 **Extensible**: Easy to add new datasets or backbones
- 🐛 **Bug-free**: Fixed the negative loss bug from the original notebook
- 📊 **Multiple datasets**: CIFAR-100 and ImageNette included

---

## 🚀 Quick Start

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

# Train on CIFAR-100
python scripts/train.py --config configs/cifar100.yaml

# Train with custom settings
python scripts/train.py --epochs 100 --batch-size 64 --lr 0.001
```

That's it! Your model will train and save checkpoints to `./checkpoints/`.

---

## 📖 Basic Usage

### Training

**Simple training:**
```bash
python scripts/train.py
```

**With configuration file:**
```bash
python scripts/train.py --config configs/default.yaml
```

**Override parameters:**
```bash
python scripts/train.py \
    --config configs/cifar100.yaml \
    --batch-size 64 \
    --epochs 200 \
    --lr 0.0005
```

**Resume from checkpoint:**
```bash
python scripts/train.py --resume checkpoints/checkpoint_latest.pth
```

### Using in Python

```python
from dino import DinoConfig, DinoTrainer
from dino.data import create_dataloaders
from dino.models import get_backbone, get_projection_head, DinoModel
from dino.loss import DinoLoss
import torch

# Load configuration
config = DinoConfig.from_yaml('configs/default.yaml')

# Create data loaders
train_loader, val_loader, _ = create_dataloaders(config)

# Create student model
backbone = get_backbone('resnet18')
projection = get_projection_head(input_dim=512, output_dim=2048)
student = DinoModel(backbone, projection)

# Create teacher (copy of student)
teacher = DinoModel(
    get_backbone('resnet18'),
    get_projection_head(input_dim=512, output_dim=2048)
)
teacher.load_state_dict(student.state_dict())

# Setup training
loss_fn = DinoLoss(out_dim=2048)
optimizer = torch.optim.AdamW(student.parameters(), lr=0.001)

trainer = DinoTrainer(
    config=config,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader
)

# Train
trainer.train(num_epochs=100)
```

---

## 📁 Project Structure

```
DinoImpl/
├── src/dino/              # Main package
│   ├── config/           # Configuration management
│   ├── data/             # Data loading & augmentation
│   ├── models/           # Neural network models
│   ├── loss/             # Loss functions
│   ├── training/         # Training logic
│   └── utils/            # Utilities (checkpointing, logging, etc.)
│
├── configs/              # YAML configuration files
│   ├── default.yaml     # ImageNette config
│   └── cifar100.yaml    # CIFAR-100 config
│
├── scripts/              # Executable scripts
│   └── train.py         # Main training script
│
├── notebooks/            # Jupyter notebooks
│   └── archive/         # Original notebook
│
├── checkpoints/          # Saved model checkpoints (auto-created)
├── logs/                 # Training logs (auto-created)
└── data/                 # Datasets (auto-downloaded)
```

---

## ⚙️ Configuration

All settings are in YAML files. Here's what you can configure:

**Data:**
- Dataset (cifar100, imagenette)
- Batch size
- Number of workers
- Train/val/test splits

**Augmentation:**
- Crop sizes (global: 224×224, local: 96×96)
- Number of local crops (default: 6)
- Color jitter, blur, solarization parameters

**Model:**
- Backbone (resnet18, resnet34, resnet50, etc.)
- Projection dimensions
- Weight normalization

**Training:**
- Number of epochs
- Learning rate
- Teacher momentum (EMA)
- Gradient clipping

**Example config:**
```yaml
data:
  dataset: imagenette
  batch_size: 32

model:
  backbone: resnet18
  projection_output_dim: 2048

training:
  num_epochs: 100
  teacher_momentum: 0.996

optimizer:
  lr: 0.001
  weight_decay: 0.04
```

See `configs/default.yaml` for all options.

---

## 🎓 How It Works (Simple Explanation)

1. **Multi-Crop Augmentation**: Takes one image, creates 8 different views (2 large, 6 small)
2. **Student Network**: Looks at all 8 views and makes predictions
3. **Teacher Network**: Only looks at the 2 large views
4. **Loss**: Student tries to match teacher's predictions
5. **EMA Update**: Teacher slowly copies student's weights
6. **Centering**: Prevents the model from always predicting the same thing

**Why this works:** The student has to learn robust features that work across different views, scales, and augmentations.

---

## 📊 Datasets

### Supported Datasets

**ImageNette** (default)
- Subset of ImageNet with 10 classes
- ~9,500 images
- Higher resolution (224×224)

**CIFAR-100**
- 100 classes of 32×32 images
- 50,000 training images
- Good for quick experiments

### Adding Your Own Dataset

1. Add it to `src/dino/data/datasets.py`
2. Create a config file in `configs/`
3. Run training!

See `ARCHITECTURE.md` for details.

---

## 🔍 Monitoring Training

**Console output:**
```
[INFO] Epoch 1/100
Epoch 1/100: 100%|████████| 208/208 [02:15<00:00]  loss: 8.2341, momentum: 0.9960
[INFO] Train Epoch 1 - loss: 8.2145, momentum: 0.9960
```

**TensorBoard** (if enabled):
```bash
tensorboard --logdir logs/
```

**Check checkpoints:**
```bash
ls -lh checkpoints/
```

---

## 💾 Checkpoints

Checkpoints are saved automatically and include:
- Student and teacher weights
- Optimizer state
- Training epoch and iteration
- Loss center for stability
- Configuration used

**Files created:**
- `checkpoint_epoch_XXXX.pth` - Checkpoint for specific epoch
- `checkpoint_latest.pth` - Most recent checkpoint
- `checkpoint_best.pth` - Best performing checkpoint (if enabled)

**Resume training:**
```bash
python scripts/train.py --resume checkpoints/checkpoint_latest.pth
```

---

## 🐛 Troubleshooting

### Loss is NaN or not decreasing
- **Check learning rate**: Try reducing it (0.0001 - 0.001)
- **Check batch size**: Increase if possible (32-64 minimum)
- **Verify data**: Make sure images load correctly

### Out of memory
- **Reduce batch size**: Try 16 or 8
- **Reduce number of local crops**: Use 4 instead of 6
- **Use smaller images**: Try CIFAR-100 first

### Slow training
- **Increase num_workers**: Set to number of CPU cores
- **Enable pin_memory**: Set to `true` in config
- **Check GPU usage**: `nvidia-smi` should show high utilization

### Loss is negative
This was a bug in the original notebook - **it's fixed in this implementation**! The loss should always be positive.

---

## 📚 Learn More

- **ARCHITECTURE.md**: Deep dive into code structure and design decisions
- **Original Paper**: [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- **Official Implementation**: [facebookresearch/dino](https://github.com/facebookresearch/dino)

---

## 🤝 Contributing

Found a bug? Want to add a feature? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 Citation

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

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Original DINO paper by Facebook AI Research
- PyTorch team for the framework
- The self-supervised learning community

---

**Questions?** Check `ARCHITECTURE.md` for technical details or open an issue!
