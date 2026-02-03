# Extending DINO

Guide to adding custom backbones, datasets, and evaluation metrics.

---

## Adding a New Backbone

The codebase already includes two backbone implementations as examples:

1. **ResNet** (`src/dino/models/backbone/resnet.py`): Uses torchvision
2. **DINO ViT** (`src/dino/models/backbone/vit.py`): Uses HuggingFace transformers

### Step 1: Create Backbone Class

Create a new file in `src/dino/models/backbone/`:

```python
# src/dino/models/backbone/efficientnet.py
from .backbone import BackboneBase
import timm

class EfficientNetBackbone(BackboneBase):
    """EfficientNet backbone using timm."""

    VARIANTS = {
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
    }

    def __init__(self, variant='efficientnet_b0', pretrained=False):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

        self.model = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        self._output_dim = self.VARIANTS[variant]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x):
        return self.model(x)
```

### Step 2: Update Factory Function

Edit `src/dino/models/backbone/backbone.py`:

```python
def get_backbone(name: str, pretrained: bool = False) -> BackboneBase:
    """Factory function for creating backbones."""

    if name.startswith('resnet'):
        from .resnet import ResnetBackboneDino
        return ResnetBackboneDino(name, pretrained)

    elif name.startswith('dino_vit'):
        from .vit import DinoBackbone
        return DinoBackbone(name, pretrained)

    elif name.startswith('efficientnet'):
        from .efficientnet import EfficientNetBackbone
        return EfficientNetBackbone(name, pretrained)

    else:
        raise ValueError(f"Unknown backbone: {name}")
```

### Step 3: Export in `__init__.py`

Edit `src/dino/models/backbone/__init__.py`:

```python
from .backbone import BackboneBase, get_backbone
from .resnet import ResnetBackboneDino
from .vit import DinoBackbone
from .efficientnet import EfficientNetBackbone

__all__ = [
    'BackboneBase',
    'get_backbone',
    'ResnetBackboneDino',
    'DinoBackbone',
    'EfficientNetBackbone',
]
```

### Step 4: Use in Config

```yaml
model:
  backbone: efficientnet_b0
  pretrained_backbone: true
```

---

## Adding a New Dataset

### Step 1: Add to `get_dataset()`

Edit `src/dino/data/datasets.py`:

```python
def get_dataset(
    name: str,
    path: str,
    transform,
    split: str = 'train'
) -> Dataset:
    """Factory function for creating datasets."""

    if name == 'imagenette':
        return Imagenette(root=path, split=split, transform=transform)

    elif name == 'imagenet100':
        return get_imagenet100_dataset(path, split, transform)

    elif name == 'cifar10':
        return torchvision.datasets.CIFAR10(
            root=path,
            train=(split == 'train'),
            transform=transform,
            download=True
        )

    elif name == 'custom_folder':
        # ImageFolder for any folder of images
        split_path = os.path.join(path, split)
        return torchvision.datasets.ImageFolder(
            root=split_path,
            transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {name}")
```

### Step 2: Create Config File

Create `configs/cifar10.yaml`:

```yaml
data:
  dataset: cifar10
  data_path: ./data
  batch_size: 128

augmentation:
  global_crop_size: 32    # CIFAR images are 32x32
  local_crop_size: 16
  global_crop_scale: [0.5, 1.0]
  local_crop_scale: [0.1, 0.5]

model:
  backbone: resnet18
  projection_output_dim: 1024
```

### Step 3: Run Training

```bash
python scripts/train.py --config configs/cifar10.yaml
```

---

## Adding Evaluation Metrics

### Step 1: Create Evaluator

Create `src/dino/evaluation/knn.py`:

```python
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, Dict

class KNNEvaluator:
    """K-Nearest Neighbors evaluation for self-supervised features."""

    def __init__(self, model, k: int = 20, device: str = 'cuda'):
        self.model = model
        self.k = k
        self.device = device

    @torch.no_grad()
    def extract_features(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from a dataloader."""
        self.model.eval()
        features_list = []
        labels_list = []

        for images, labels in dataloader:
            images = images.to(self.device)

            # Get backbone features (not projections)
            features, _ = self.model(images, return_backbone_features=True)
            features = features.cpu().numpy()

            features_list.append(features)
            labels_list.append(labels.numpy())

        return np.concatenate(features_list), np.concatenate(labels_list)

    def evaluate(
        self,
        train_loader,
        val_loader
    ) -> Dict[str, float]:
        """Run KNN evaluation."""
        # Extract features
        train_features, train_labels = self.extract_features(train_loader)
        val_features, val_labels = self.extract_features(val_loader)

        # Normalize features
        train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
        val_features = val_features / np.linalg.norm(val_features, axis=1, keepdims=True)

        # Fit KNN
        knn = KNeighborsClassifier(n_neighbors=self.k, metric='cosine')
        knn.fit(train_features, train_labels)

        # Evaluate
        accuracy = knn.score(val_features, val_labels)

        return {
            'knn_accuracy': accuracy,
            'k': self.k
        }
```

### Step 2: Export in `__init__.py`

Edit `src/dino/evaluation/__init__.py`:

```python
from .knn import KNNEvaluator

__all__ = ['KNNEvaluator']
```

### Step 3: Add to Trainer (Optional)

```python
# In trainer.py
class DinoTrainer:
    def __init__(self, ..., evaluator=None):
        self.evaluator = evaluator

    def validate_epoch(self, epoch):
        if self.evaluator:
            # Create eval dataloaders (without multi-crop)
            eval_train_loader = create_eval_dataloader(self.train_loader.dataset)
            eval_val_loader = create_eval_dataloader(self.val_loader.dataset)

            metrics = self.evaluator.evaluate(eval_train_loader, eval_val_loader)
            self.logger.info(f"KNN Accuracy: {metrics['knn_accuracy']:.4f}")
            return metrics
        return {}
```

### Step 4: Use Evaluator

```python
from dino.evaluation import KNNEvaluator

# After training
evaluator = KNNEvaluator(model=teacher, k=20, device='cuda')
metrics = evaluator.evaluate(train_loader, val_loader)
print(f"KNN Accuracy: {metrics['knn_accuracy']:.2%}")
```

---

## Adding Custom Augmentations

### Step 1: Create Custom Transform

```python
# src/dino/data/custom_transforms.py
import torchvision.transforms as T

class CustomAugmentation:
    """Custom augmentation for specific dataset."""

    def __init__(self, p: float = 0.5):
        self.transform = T.Compose([
            T.RandomApply([
                T.GaussianBlur(kernel_size=23)
            ], p=p),
            # Add more augmentations
        ])

    def __call__(self, img):
        return self.transform(img)
```

### Step 2: Integrate with DINOTransform

```python
# In DINOTransform
class DINOTransform:
    def __init__(self, ..., custom_aug=None):
        self.custom_aug = custom_aug

    def __call__(self, img):
        if self.custom_aug:
            img = self.custom_aug(img)
        # Continue with standard transforms
        ...
```

---

## Adding Custom Loss Functions

```python
# src/dino/loss/custom_loss.py
import torch
import torch.nn as nn

class CustomDinoLoss(nn.Module):
    """Custom DINO loss with additional regularization."""

    def __init__(self, base_loss, reg_weight: float = 0.1):
        super().__init__()
        self.base_loss = base_loss
        self.reg_weight = reg_weight

    def forward(self, student_output, teacher_output):
        # Base DINO loss
        loss = self.base_loss(student_output, teacher_output)

        # Add regularization
        reg = self.compute_regularization(student_output)
        total_loss = loss + self.reg_weight * reg

        return total_loss

    def compute_regularization(self, output):
        # Example: entropy regularization
        probs = torch.softmax(output, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return -entropy.mean()  # Maximize entropy
```

---

## See Also

- [Models](../guides/models.md) - Existing model architectures
- [Data Pipeline](../guides/data-pipeline.md) - Data loading details
- [API Reference](../api/models.md) - Full API documentation
