# Data API Reference

API documentation for the data module.

---

## DINOTransform

::: dino.data.DINOTransform
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - __call__
        - from_config

---

## Datasets

### get_dataset

::: dino.data.get_dataset
    options:
      show_root_heading: true
      show_source: true

---

## DataLoaders

### create_dataloaders

::: dino.data.create_dataloaders
    options:
      show_root_heading: true
      show_source: true

### collate_multi_crop

::: dino.data.collate_multi_crop
    options:
      show_root_heading: true
      show_source: true

---

## Usage Examples

### Creating DataLoaders from Config

```python
from dino.config import DinoConfig
from dino.data import create_dataloaders

config = DinoConfig.from_yaml('configs/default.yaml')
train_loader, val_loader, test_loader = create_dataloaders(config)

for views, labels in train_loader:
    # views: list of 8 tensors, each [batch, 3, H, W]
    # labels: [batch]
    print(f"Number of views: {len(views)}")
    print(f"Global view shape: {views[0].shape}")
    print(f"Local view shape: {views[2].shape}")
    break
```

### Using DINOTransform

```python
from dino.data import DINOTransform
from PIL import Image

# Create transform
transform = DINOTransform(
    num_local_views=6,
    global_crop_size=224,
    local_crop_size=96
)

# Apply to image
image = Image.open('example.jpg')
views = transform(image)

print(f"Number of views: {len(views)}")  # 8
print(f"Global view shape: {views[0].shape}")  # [3, 224, 224]
print(f"Local view shape: {views[2].shape}")  # [3, 96, 96]
```

### Creating Transform from Config

```python
from dino.config import DinoConfig
from dino.data import DINOTransform

config = DinoConfig.from_yaml('configs/default.yaml')
transform = DINOTransform.from_config(config.augmentation)
```

### Getting Datasets

```python
from dino.data import get_dataset, DINOTransform

transform = DINOTransform(num_local_views=6)

# ImageNette
dataset = get_dataset(
    name='imagenette',
    path='./data',
    transform=transform,
    split='train'
)

# ImageNet100
dataset = get_dataset(
    name='imagenet100',
    path='./data/imagenet100',
    transform=transform,
    split='train'
)
```

### Custom Collate Function

```python
from dino.data import collate_multi_crop
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_multi_crop,
    num_workers=4,
    pin_memory=True
)
```
