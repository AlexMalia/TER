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


### Augmentation Details 

The values explained here are based on the original DINO paper and the one integrated into the default configuration. You can customize these values in the config file.

#### Crop Sizes and Scales

| View Type | Count | Crop Size | Scale Range | Special |
|-----------|-------|-----------|-------------|---------|
| Global 1 | 1 | 224×224 | 0.4 - 1.0 | Standard augmentation |
| Global 2 | 1 | 224×224 | 0.4 - 1.0 | + Solarization |
| Local | 6 | 96×96 | 0.05 - 0.4 | Aggressive cropping |

This pipeline generates multiple "views" of the same image to support self-supervised learning:

* **Global Views (224×224):** 
    * Large crop, random crop size between 40% and 100% of the original image. 
    * Aim to capture the primary semantic context, ensuring the main object is mostly visible.
    * Resized to 224×224.

* **Local Views (96×96):** 
    * Small crop, random crop size between 5% and 40% of the original image. 
    * Focus on fine-grained textures and local features, forcing the model to learn "part-to-whole" relationships.
    * Resized to 96x96.

#### Other Augmentations


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
- **Resolution**: 
    - Full size: 500px
    - Medium size: 320px
- **Download**: Automatic via torchvision

```yaml
# configs/default.yaml
data:
  dataset: imagenette
  data_path: ./data
```

#### ImageNet100

**WIP**: The informations below need to be taken with caution.

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
uv add kaggle

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

**Exemple**
If you have a batch of 4 images, each with 8 views, the collate function will transform:

```
[
  [g1_img1, g2_img1, l1_img1, ..., l6_img1],
  [g1_img2, g2_img2, l1_img2, ..., l6_img2],
  [g1_img3, g2_img3, l1_img3, ..., l6_img3],
  [g1_img4, g2_img4, l1_img4, ..., l6_img4]
]
```

into:

```
[
  [g1_img1, g1_img2, g1_img3, g1_img4],  # Global view 1 batch
  [g2_img1, g2_img2, g2_img3, g2_img4],  # Global view 2 batch
  [l1_img1, l1_img2, l1_img3, l1_img4],  # Local view 1 batch
  ...
  [l6_img1, l6_img2, l6_img3, l6_img4]   # Local view 6 batch
]
```

Shapes before collate:
- Each view tensor: `[3, H, W]` (e.g., `[3, 224, 224]` for global, `[3, 96, 96]` for local)

Shapes after collate:
- Global view batches: `[batch_size, 3, 224, 224]`
- Local view batches: `[batch_size, 3, 96, 96]`

### DataLoader Configuration

```yaml
data:
  batch_size: 32
  num_workers: 4
  pin_memory: true
```

---


## Complete Data Table

### Transformation

| Step | Shape | Description |
|------|-------|-------------|
| **Input** | `[3, H, W]` | Original RGB image from dataset |
| **After DINOTransform** | `List[8 tensors]` | Multi-crop augmentation applied |

**Output Details:**
- **2 Global views:** `[3, 224, 224]` each large crops covering 40-100% of original image
- **6 Local views:** `[3, 96, 96]` each small crops covering 5-40% of original image

**Result:** 1 image → 8 augmented views

---

### Batch Collation

**Input Structure**
```
List[Tuple[List[8 tensors], label]]
│
├─ Sample 0: ([8 view tensors : 3, H, W], label_0)
├─ Sample 1: ([8 view tensors : 3, H, W], label_1)
├─ Sample 2: ([8 view tensors : 3, H, W], label_2)
└─ ... (batch_size samples total)
```

**Type:** `List[Tuple[List[Tensor], int]]`  
**Length:** `batch_size` (e.g., 32, 64, 128)  
**Per sample:** 8 tensors (2 global + 6 local) + 1 integer label

**Output Structure**
```
Tuple[List[8 batched tensors], labels_tensor]
│
├─ views_batch: [
│   ├─ [0] Global view 1:  [batch_size, 3, 224, 224]
│   ├─ [1] Global view 2:  [batch_size, 3, 224, 224]
│   ├─ [2] Local view 1:   [batch_size, 3, 96, 96]
│   ├─ [3] Local view 2:   [batch_size, 3, 96, 96]
│   ├─ [4] Local view 3:   [batch_size, 3, 96, 96]
│   ├─ [5] Local view 4:   [batch_size, 3, 96, 96]
│   ├─ [6] Local view 5:   [batch_size, 3, 96, 96]
│   └─ [7] Local view 6:   [batch_size, 3, 96, 96]
│   ]
│
└─ labels_batch: [batch_size]
```

**Type:** `Tuple[List[Tensor], Tensor]`

---

### Example batch size 32

**Before Collation**
```python
# 32 individual samples
[
    ([tensor, tensor, ..., tensor], 5),   # Image 1: 8 views + label
    ([tensor, tensor, ..., tensor], 3),   # Image 2: 8 views + label
    ...                                    # 30 more samples
]
```

**After Collation**
```python
(
    [  # 8 batched tensors
        torch.Size([32, 3, 224, 224]),  # Global view 1 (all 32 images)
        torch.Size([32, 3, 224, 224]),  # Global view 2 (all 32 images)
        torch.Size([32, 3, 96, 96]),    # Local view 1 (all 32 images)
        torch.Size([32, 3, 96, 96]),    # Local view 2 (all 32 images)
        torch.Size([32, 3, 96, 96]),    # Local view 3 (all 32 images)
        torch.Size([32, 3, 96, 96]),    # Local view 4 (all 32 images)
        torch.Size([32, 3, 96, 96]),    # Local view 5 (all 32 images)
        torch.Size([32, 3, 96, 96]),    # Local view 6 (all 32 images)
    ],
    torch.Size([32])  # All 32 labels: [5, 3, 7, 1, ...]
)
```

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

2. **Config file** `configs/custom_dataset.yaml`:

```yaml
data:
  dataset: custom_dataset
  data_path: ./data
  batch_size: 32
```

3. **Run training**:

```bash
uv run scripts/train.py --config configs/custom_dataset.yaml
```

## See Also

- [Configuration](configuration.md) - Full configuration reference
- [Models](models.md) - Backbone architectures
- [API Reference: Data](../api/data.md) - API documentation
