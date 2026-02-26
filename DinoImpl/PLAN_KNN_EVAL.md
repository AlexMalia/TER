# Plan: Add KNN Classifier Evaluation to DINO Training

## Context

The DINO paper uses a k-nearest-neighbor (KNN) classifier to monitor whether learned representations
separate classes without supervision. This plan adds KNN evaluation using sklearn's
`KNeighborsClassifier` (cosine metric) on L2-normalized teacher backbone features.

Two modes:
1. **Periodic evaluation during training** — runs every N epochs inside `DinoTrainer.train()`, logs to console + wandb.
2. **Standalone evaluation script** — `scripts/evaluate_knn.py` loads any checkpoint and runs KNN.

---

## Files to Modify / Create

| File | Action |
|------|--------|
| `src/dino/evaluation/knn.py` | **Create** – KNNEvaluator class using sklearn |
| `src/dino/evaluation/__init__.py` | **Edit** – export `KNNEvaluator` |
| `src/dino/config/config.py` | **Edit** – add `EvaluationConfig` dataclass |
| `src/dino/data/dataloaders.py` | **Edit** – add `create_knn_dataloaders()` |
| `src/dino/training/trainer.py` | **Edit** – integrate periodic KNN eval |
| `scripts/train.py` | **Edit** – create evaluator and pass to trainer |
| `scripts/evaluate_knn.py` | **Create** – standalone KNN evaluation script |
| `configs/*.yaml` | **Edit** – add `evaluation_config` section to all configs |

All paths are relative to `DinoImpl/`.

---

## Step-by-Step Implementation

### Step 1 — Add `EvaluationConfig` to `config.py`

**File:** `src/dino/config/config.py`

Add a new dataclass after `LoggingConfig` (line 144). All fields have defaults so existing YAMLs
that omit this section keep working:

```python
@dataclass
class EvaluationConfig:
    use_knn_eval: bool = True
    eval_every_n_epochs: int = 10
    knn_k: int = 20
```

Update `_CONFIG_CLASSES` (line 146) to include:
```python
'evaluation_config': EvaluationConfig,
```

Add to `DinoConfig` (line 176) with a default factory so it is optional in existing code:
```python
evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
```

> **Note:** `default_factory` is required because `EvaluationConfig` is a mutable dataclass.
> The `field` import is already present at line 3.

---

### Step 2 — Create `src/dino/evaluation/knn.py`

**File:** `src/dino/evaluation/knn.py` *(new file)*

Key design choices:
- Features are L2-normalized before sklearn so cosine distance = dot product.
- `algorithm='brute'` is mandatory for sklearn's cosine metric.
- Top-5 is computed via `predict_proba` and skipped when fewer than 5 classes exist.
- `model.get_backbone_features(x)` is called — bypasses the projection head entirely
  (line 59 of `src/dino/models/dino_model.py`).

```python
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import logging

logger = logging.getLogger(__name__)


class KNNEvaluator:
    def __init__(self, k: int = 20):
        self.k = k

    @torch.no_grad()
    def extract_features(self, loader, model, device) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        features, labels = [], []
        for images, targets in loader:
            images = images.to(device)
            feats = model.get_backbone_features(images)
            feats = F.normalize(feats, dim=-1)
            features.append(feats.cpu())
            labels.append(targets)
        return torch.cat(features).numpy(), torch.cat(labels).numpy()

    def evaluate(self, train_loader, val_loader, model, device) -> dict:
        logger.info("KNN: extracting train features...")
        train_feats, train_labels = self.extract_features(train_loader, model, device)

        logger.info("KNN: extracting val features...")
        val_feats, val_labels = self.extract_features(val_loader, model, device)

        logger.info(f"KNN: fitting k={self.k} on {len(train_feats)} samples...")
        knn = KNeighborsClassifier(
            n_neighbors=self.k,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1,
        )
        knn.fit(train_feats, train_labels)

        top1 = float((knn.predict(val_feats) == val_labels).mean() * 100)

        top5 = None
        if len(knn.classes_) >= 5:
            proba = knn.predict_proba(val_feats)
            top5_indices = np.argsort(proba, axis=1)[:, -5:]
            top5_class_labels = knn.classes_[top5_indices]
            top5 = float(
                np.mean([val_labels[i] in top5_class_labels[i]
                         for i in range(len(val_labels))]) * 100
            )

        result = {"knn_top1": top1, "knn_k": self.k}
        if top5 is not None:
            result["knn_top5"] = top5
        return result
```

---

### Step 3 — Update `src/dino/evaluation/__init__.py`

**File:** `src/dino/evaluation/__init__.py`

Replace the current (empty) file with:

```python
from .knn import KNNEvaluator
__all__ = ["KNNEvaluator"]
```

---

### Step 4 — Add `create_knn_dataloaders()` to `dataloaders.py`

**File:** `src/dino/data/dataloaders.py`

Append after the existing `create_dataloaders()` function (after line 158).

Key design choices:
- Uses standard eval transforms (Resize → CenterCrop → ToTensor → Normalize); no multi-crop.
- Calls the same `create_train_val_test_splits()` with the same seed as training, so the splits
  are identical — no data leakage from the test set.
- No `collate_multi_crop`; the loader returns plain `(image, label)` tuples.
- `shuffle=False` on both splits (order doesn't matter for KNN fitting).

```python
def create_knn_dataloaders(
    data_config: DataConfig,
    augmentation_config: AugmentationConfig,
    batch_size: int = 256,
) -> tuple:
    from torchvision import transforms as T
    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(augmentation_config.global_crop_size),
        T.ToTensor(),
        T.Normalize(augmentation_config.normalize_mean, augmentation_config.normalize_std),
    ])
    dataset = get_dataset(
        data_config.dataset, data_config.data_path,
        transform=eval_transform, download=True, train=True
    )
    train_ds, val_ds, _ = create_train_val_test_splits(
        dataset,
        train_split=data_config.train_split,
        val_split=data_config.val_split,
        seed=data_config.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.do_pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.do_pin_memory,
    )
    return train_loader, val_loader
```

---

### Step 5 — Integrate KNN into `DinoTrainer` (`trainer.py`)

**File:** `src/dino/training/trainer.py`

**5a. Extend `__init__` signature** (after line 76, `val_loader` parameter):

```python
def __init__(
    self,
    config: DinoConfig,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LRScheduler],
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: str = 'cuda',
    knn_evaluator=None,        # new
    knn_train_loader=None,     # new
    knn_val_loader=None,       # new
):
```

Store as instance attributes in `__init__` body:
```python
self.knn_evaluator = knn_evaluator
self.knn_train_loader = knn_train_loader
self.knn_val_loader = knn_val_loader
```

**5b. Add KNN eval block inside `train()`** after the epoch-level wandb log (after line 229),
before the checkpoint save block:

```python
eval_cfg = self.config.evaluation_config
if (self.knn_evaluator is not None
        and eval_cfg.use_knn_eval
        and epoch % eval_cfg.eval_every_n_epochs == 0):
    knn_metrics = self.knn_evaluator.evaluate(
        self.knn_train_loader, self.knn_val_loader,
        self.teacher, self.device
    )
    top1 = knn_metrics["knn_top1"]
    top5 = knn_metrics.get("knn_top5")
    msg = f"KNN Eval (k={knn_metrics['knn_k']}) — Top-1: {top1:.2f}%"
    if top5 is not None:
        msg += f", Top-5: {top5:.2f}%"
    logger.info(msg)

    if wandb is not None and wandb.run is not None:
        log_dict = {
            "epoch": epoch,
            "eval/knn_top1": top1,
            "eval/knn_k": knn_metrics["knn_k"],
        }
        if top5 is not None:
            log_dict["eval/knn_top5"] = top5
        wandb.log(log_dict)
```

---

### Step 6 — Update `scripts/train.py`

**File:** `scripts/train.py`

**6a.** Add imports after the existing import block:

```python
from dino.data.dataloaders import create_knn_dataloaders
from dino.evaluation import KNNEvaluator
```

**6b.** After the training dataloaders are created (after line 94), add:

```python
knn_evaluator = knn_train_loader = knn_val_loader = None
if config.evaluation_config.use_knn_eval:
    logger.info("Creating KNN evaluation dataloaders...")
    knn_train_loader, knn_val_loader = create_knn_dataloaders(
        config.data_config, config.augmentation_config
    )
    knn_evaluator = KNNEvaluator(k=config.evaluation_config.knn_k)
    logger.info(
        f"KNN evaluator ready (k={config.evaluation_config.knn_k}, "
        f"every {config.evaluation_config.eval_every_n_epochs} epochs)"
    )
```

**6c.** Pass the new arguments to `DinoTrainer` (around line 144):

```python
trainer = DinoTrainer(
    config=config,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    scheduler=optim_scheduler,
    loss_fn=dino_loss,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    knn_evaluator=knn_evaluator,       # new
    knn_train_loader=knn_train_loader, # new
    knn_val_loader=knn_val_loader,     # new
)
```

**6d.** Add `eval/*` metric axis to the wandb `define_metric` block (after line 192):

```python
wandb.define_metric("eval/*", step_metric="epoch")
```

---

### Step 7 — Create `scripts/evaluate_knn.py` (standalone script)

**File:** `scripts/evaluate_knn.py` *(new file)*

Loads any checkpoint and runs KNN independently of the training loop.
Uses the same `load_checkpoint` utility as the trainer.

```python
#!/usr/bin/env python3
"""Standalone KNN evaluation from a DINO checkpoint."""

import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dino.config.config import DinoConfig
from dino.models import DinoModel
from dino.utils.checkpoint import load_checkpoint
from dino.data.dataloaders import create_knn_dataloaders
from dino.evaluation import KNNEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="KNN evaluation from DINO checkpoint")
    p.add_argument("--config",     required=True, help="Path to config YAML")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--k",          type=int, default=20, help="Number of neighbors")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    config = DinoConfig.from_yaml_and_args(args.config, args)
    device = args.device if torch.cuda.is_available() else "cpu"

    teacher = DinoModel.from_config(config.model_config)
    for p in teacher.parameters():
        p.requires_grad = False
    load_checkpoint(
        args.checkpoint, teacher=teacher, device=device,
        student=None, optimizer=None, dino_loss=None,
    )
    teacher.to(device)

    knn_train_loader, knn_val_loader = create_knn_dataloaders(
        config.data_config, config.augmentation_config, batch_size=args.batch_size
    )
    metrics = KNNEvaluator(k=args.k).evaluate(knn_train_loader, knn_val_loader, teacher, device)

    print(f"\nKNN Results (k={metrics['knn_k']}):")
    print(f"  Top-1 accuracy: {metrics['knn_top1']:.2f}%")
    if "knn_top5" in metrics:
        print(f"  Top-5 accuracy: {metrics['knn_top5']:.2f}%")


if __name__ == "__main__":
    main()
```

---

### Step 8 — Update YAML configs

**Files:** all `configs/*.yaml` that use the `_config` key format (i.e., `imagenette.yaml`,
`imagenet100.yaml`, `imagenet100-resnet.yaml`, `kaggle-imagenet100.yaml`, `kaggle-imagenette.yaml`,
`default_debug.yaml`).

Add at the end of each file:

```yaml
evaluation_config:
  use_knn_eval: true
  eval_every_n_epochs: 10
  knn_k: 20
```

> **Note:** `default.yaml` and `example.yaml` use the old short-key format (`data:`,
> `augmentation:`, etc.) and are not parsed by the current loader. They do not need updating.

---

## Reused Existing Code

| Utility | Location | Purpose |
|---------|----------|---------|
| `DinoModel.get_backbone_features(x)` | `src/dino/models/dino_model.py:59` | Feature extraction (no projection head) |
| `create_train_val_test_splits(...)` | `src/dino/data/datasets.py:103` | Produce identical splits in KNN loaders |
| `get_dataset(...)` | `src/dino/data/datasets.py:15` | Load dataset with eval transform |
| `AugmentationConfig.normalize_mean/std` | `src/dino/config/config.py:59` | Eval transform normalization |
| `AugmentationConfig.global_crop_size` | `src/dino/config/config.py:33` | CenterCrop size |
| `load_checkpoint(...)` | `src/dino/utils/checkpoint.py` | Load teacher weights in standalone script |
| `wandb.log()` pattern | `src/dino/training/trainer.py:223` | Consistent with existing logging |

---

## Verification Steps

### 1. Training with KNN (fast sanity check)

```bash
uv run python DinoImpl/scripts/train.py \
  --config DinoImpl/configs/imagenette.yaml \
  --epochs 2
```

Expected: KNN Top-1 printed at epoch 1 and 2. Even with random weights, should be well above
random chance (~10%) once the backbone begins learning.

### 2. Standalone evaluation on a saved checkpoint

```bash
uv run python DinoImpl/scripts/evaluate_knn.py \
  --config DinoImpl/configs/imagenette.yaml \
  --checkpoint DinoImpl/checkpoints/checkpoint_latest.pth \
  --k 20
```

### 3. Wandb

`eval/knn_top1` and `eval/knn_top5` should appear with epoch as the x-axis in the dashboard.

### 4. sklearn dependency

Verify sklearn is in `DinoImpl/pyproject.toml`. Add `scikit-learn` if missing.

---

## Notes

- `EvaluationConfig` uses `default_factory` so existing code that builds `DinoConfig` directly
  (e.g., in tests) does not need updating.
- KNN runs on the **teacher** model only — teacher features are better calibrated than student's
  because the teacher is updated via EMA.
- Feature extraction is done with `torch.no_grad()` and the model set to `.eval()`, so batch-norm
  statistics are stable.
- The `eval_every_n_epochs` condition uses `epoch % N == 0`, which triggers at epochs
  N, 2N, 3N, … (not epoch 0).
