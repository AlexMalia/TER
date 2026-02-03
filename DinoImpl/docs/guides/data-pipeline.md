# Data Pipeline

This guide explains the DINO data pipeline: augmentations, datasets, and dataloaders.

---

## Overview

The data pipeline follows this flow:

```
Image Dataset
    ↓
DINOTransform (multi-crop augmentation)
    ↓
DataLoader (batch + custom collate)
    ↓
[8 views per image: 2 global + 6 local]
```

---

## Multi-Crop Transform

The `DINOTransform` class creates multiple augmented views of each image.

### Architecture

```python
class DINOTransform:
    def __init__(self, num_local_views=6, ...):
        # Global view 1: Standard augmentation
        self.global_t1 = Compose([
            RandomResizedCrop(224, scale=(0.4, 1.0)),
            ColorJitter + GaussianBlur
        ])

        # Global view 2: Add solarization
        self.global_t2 = Compose([
            ..., RandomSolarize(...)
        ])

        # Local views: Smaller crops
        self.local_transform = Compose([
            RandomResizedCrop(96, scale=(0.05, 0.4)),
            ...
        ])

    def __call__(self, img) -> List[Tensor]:
        return [
            self.global_t1(img),
            self.global_t2(img),
            *[self.local_transform(img) for _ in range(6)]
        ]
```

### View Types

| View Type | Count | Crop Size | Scale Range | Special |
|-----------|-------|-----------|-------------|---------|
| Global 1 | 1 | 224×224 | 0.4 - 1.0 | Standard augmentation |
| Global 2 | 1 | 224×224 | 0.4 - 1.0 | + Solarization |
| Local | 6 | 96×96 | 0.05 - 0.4 | Aggressive cropping |

### Augmentation Details

**Color Jitter**:
- Brightness: 0.4
- Contrast: 0.4
- Saturation: 0.2
- Hue: 0.1

**Gaussian Blur**:
- Applied with probability 0.5 (global views)
- Kernel size proportional to crop size

**Solarization**:
- Applied only to global view 2
- Probability: 0.2

### Using DINOTransform

```python
from dino.data import DINOTransform

# From config
transform = DINOTransform.from_config(config.augmentation)

# Manual creation
transform = DINOTransform(
    num_local_views=6,
    global_crop_size=224,
    local_crop_size=96,
    global_crop_scale=(0.4, 1.0),
    local_crop_scale=(0.05, 0.4)
)

# Apply to image
views = transform(image)  # Returns list of 8 tensors
```

---

## Datasets

### Supported Datasets

#### ImageNette (Default)

- **Description**: 10-class subset of ImageNet
- **Size**: ~9,500 images
- **Resolution**: 224×224
- **Download**: Automatic via torchvision

```yaml
# configs/default.yaml
data:
  dataset: imagenette
  data_path: ./data
```

#### ImageNet100

- **Description**: 100-class subset of ImageNet (from Kaggle)
- **Size**: ~130,000 training images
- **Resolution**: 224×224
- **Download**: Manual (Kaggle)

```yaml
# configs/imagenet100.yaml
data:
  dataset: imagenet100
  data_path: ./data/imagenet100
```

**Setup ImageNet100**:

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle account)
kaggle datasets download -d ambityga/imagenet100

# Extract to data directory
unzip imagenet100.zip -d ./data/imagenet100
```

The dataset has a multi-folder structure (`train.X1`, `train.X2`, `train.X3`, `train.X4`, `val.X`) that is handled automatically via `ConcatDataset`.

### Factory Function

```python
from dino.data import get_dataset

# Get dataset with transform
dataset = get_dataset(
    name='imagenette',
    path='./data',
    transform=transform,
    split='train'
)
```

---

## DataLoaders

### Creating DataLoaders

```python
from dino.data import create_dataloaders

# Create all dataloaders from config
train_loader, val_loader, test_loader = create_dataloaders(config)
```

### Custom Collate Function

Multi-crop transforms return a list of tensors, requiring a custom collate function:

```python
def collate_multi_crop(batch):
    """Transpose list-of-lists to list of batched tensors."""
    views_lists = [item[0] for item in batch]  # Each: [g1, g2, l1, ..., l6]
    labels = [item[1] for item in batch]

    # Transpose: [[g1_img1, g1_img2, ...], [g2_img1, g2_img2, ...], ...]
    views_batch = [
        torch.stack([views[i] for views in views_lists])
        for i in range(num_views)
    ]
    return views_batch, torch.tensor(labels)
```

**Why custom collate?**

- Multi-crop returns list of tensors, not single tensor
- Need to group by view type (all global1 together, all global2 together, etc.)

### DataLoader Configuration

```yaml
data:
  batch_size: 32
  num_workers: 4
  pin_memory: true
```

---

## Tensor Shapes

| Stage | Shape | Notes |
|-------|-------|-------|
| Original image | `[3, H, W]` | Single RGB image |
| After DINOTransform | `List[8 tensors]` | Each: `[3, crop_size, crop_size]` |
| After collate | `List[8 batches]` | Each: `[batch, 3, crop_size, crop_size]` |
| Global views | `[batch, 3, 224, 224]` | 2 views |
| Local views | `[batch, 3, 96, 96]` | 6 views |

---

## Adding Custom Datasets

1. **Add to `get_dataset()`** in `src/dino/data/datasets.py`:

```python
elif dataset_name == 'custom_dataset':
    return torchvision.datasets.ImageFolder(
        root=os.path.join(data_path, 'custom'),
        transform=transform
    )
```

2. **Create config file** `configs/custom_dataset.yaml`:

```yaml
data:
  dataset: custom_dataset
  data_path: ./data
  batch_size: 32
```

3. **Run training**:

```bash
python scripts/train.py --config configs/custom_dataset.yaml
```

---

## Performance Tips

- **num_workers**: Set to number of CPU cores
- **pin_memory**: Enable for GPU training
- **persistent_workers**: Keep workers alive between epochs

```yaml
data:
  num_workers: 8
  pin_memory: true
```

---

## See Also

- [Configuration](configuration.md) - Full configuration reference
- [Models](models.md) - Backbone architectures
- [API Reference: Data](../api/data.md) - API documentation
