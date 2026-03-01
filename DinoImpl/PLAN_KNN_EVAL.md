# Plan : KNN Visualization (accuracy curve + feature embeddings)

## Context

The KNN evaluator currently only logs scalar top-1/top-5 accuracies to wandb and console.
Two visualizations are needed, saved as local PNG files:
1. **KNN accuracy curve over epochs** — sparse (every N epochs), plotted via `History`
2. **Feature space embedding (t-SNE)** — scatter plot of val features colored by class

---

## Files to modify

1. `src/dino/config/config.py`
2. `src/dino/utils/history.py`
3. `src/dino/evaluation/knn.py`
4. `src/dino/training/trainer.py`
5. `configs/imagenette.yaml`
6. `kaggle/code-dataset/configs/kaggle-imagenette.yaml`

---

## Detailed changes

### 1. `src/dino/config/config.py` — Add embedding viz fields to `EvaluationConfig`

```python
@dataclass
class EvaluationConfig:
    use_knn_eval: bool
    eval_every_n_epochs: int
    knn_ks: list[int]
    knn_temperature: float
    knn_batch_size: int
    knn_embedding_viz: bool = False        # ← new
    knn_embedding_n_samples: int = 2000    # ← new (max samples for t-SNE speed)
```

---

### 2. `src/dino/utils/history.py` — Add KNN history + `plot_knn()`

#### A. `__init__`: add `knn_metrics` list
```python
self.knn_metrics: List[Dict[str, Any]] = []
```

#### B. Add `record_knn_eval(epoch, metrics)`:
```python
def record_knn_eval(self, epoch: int, metrics: Dict[str, float]) -> None:
    self.knn_metrics.append({'epoch': epoch, **metrics})
```

#### C. Add `get_knn_epochs()`:
```python
def get_knn_epochs(self) -> List[int]:
    return [r['epoch'] for r in self.knn_metrics]
```

#### D. Add `plot_knn(ax=None, save_path=None)`:
- Plots one line per k value found in `knn_metrics` (keys matching `knn_top1_k*`)
- X-axis: epoch number (sparse, so use markers: `marker='o'`)
- Y-axis: accuracy (0–100), label "KNN Top-1 Accuracy (%)"
- Returns `ax`. If `save_path` given, saves fig at 150 DPI.
- Skips silently if `self.knn_metrics` is empty.

#### E. Update `to_dict()` / `from_dict()` to persist `knn_metrics`:
```python
# to_dict:
'knn_metrics': self.knn_metrics
# from_dict:
history.knn_metrics = data.get('knn_metrics', [])
```

---

### 3. `src/dino/evaluation/knn.py` — Add `plot_embeddings()`

New method on `KNNClassifier`:

```python
def plot_embeddings(
    self,
    model: DinoModel,
    data_loader: DataLoader,
    save_path=None,
    n_samples: int = 2000,
    class_names=None,
    epoch: Optional[int] = None,
) -> "plt.Figure":
```

Implementation steps:
1. Call `self.extract_features(model, data_loader)` → `(features, labels)` tensors (already L2-normalized)
2. Sub-sample to `n_samples` if dataset is larger (random permutation)
3. Run `sklearn.manifold.TSNE(n_components=2, random_state=42).fit_transform(features.numpy())`
4. Create scatter plot with one color per class using `matplotlib`
5. Add legend using `class_names` if provided, else `"Class {i}"`
6. Title: `f"Feature Embeddings (t-SNE) — Epoch {epoch}"` if epoch provided
7. If `save_path` given, save at 150 DPI
8. Return figure

---

### 4. `src/dino/training/trainer.py` — Wire both visualizations

#### A. Inside the `if epoch % eval_every_n_epochs == 0:` block, after logging KNN scalars:

```python
# Record in history (enables persistent KNN curve)
self.history.record_knn_eval(epoch, knn_metrics)

# Save KNN accuracy curve (overwrites each time, always up-to-date)
log_dir = Path(self.config.logging_config.log_dir)
self.history.plot_knn(
    save_path=log_dir / "knn_accuracy.png"
)

# Feature embedding visualization (optional, controlled by config)
if self.config.evaluation_config.knn_embedding_viz:
    embed_dir = log_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    # Extract class names from the underlying dataset (unwrap Subset wrappers)
    class_names = None
    dataset = self.val_eval_loader.dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    class_names = getattr(dataset, 'classes', None)

    self.evaluator.plot_embeddings(
        model=self.teacher,
        data_loader=self.val_eval_loader,
        save_path=embed_dir / f"epoch_{epoch:04d}.png",
        n_samples=self.config.evaluation_config.knn_embedding_n_samples,
        class_names=class_names,
        epoch=epoch,
    )
```

#### B. Add `from pathlib import Path` import if not already present.

---

### 5. `configs/imagenette.yaml` — Add new fields to `evaluation_config`

```yaml
evaluation_config:
  use_knn_eval: true
  eval_every_n_epochs: 1
  knn_ks: [1, 5, 20]
  knn_temperature: 0.07
  knn_batch_size: 256
  knn_embedding_viz: false       # ← new
  knn_embedding_n_samples: 2000  # ← new
```

---

### 6. `kaggle/code-dataset/configs/kaggle-imagenette.yaml` — Same addition

```yaml
evaluation_config:
  use_knn_eval: true
  eval_every_n_epochs: 1
  knn_ks: [1, 5, 20]
  knn_temperature: 0.07
  knn_batch_size: 256
  knn_embedding_viz: false       # ← new
  knn_embedding_n_samples: 2000  # ← new
```

---

## Output files produced

| File | When produced |
|---|---|
| `{log_dir}/knn_accuracy.png` | Every KNN eval (overwritten) |
| `{log_dir}/embeddings/epoch_{N:04d}.png` | Every KNN eval, only if `knn_embedding_viz: true` |

---

## Verification

1. Run a short training with `eval_every_n_epochs: 1` and `num_epochs: 3`
2. Check that `logs/.../knn_accuracy.png` exists and shows a curve with one line per k
3. Enable `knn_embedding_viz: true`, re-run → check `logs/.../embeddings/epoch_0001.png` exists
4. Resume from checkpoint — KNN curve should reconstruct from saved `knn_metrics` in history

```bash
uv run python scripts/train.py --config configs/imagenette.yaml --epochs 3
```
