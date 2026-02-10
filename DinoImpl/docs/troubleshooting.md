# Troubleshooting

Solutions to common problems when using DINO.

---

## Training Issues

### Loss is NaN or Not Decreasing

**Symptoms**: Loss shows `nan` or stays flat

**Solutions**:

1. **Reduce learning rate**:
   ```yaml
   optimizer:
     lr: 0.0001  # Try lower values
   ```

2. **Check batch size** (need sufficient examples):
   ```yaml
   data:
     batch_size: 32  # Minimum recommended
   ```

3. **Verify data pipeline**:
   ```python
   for views, labels in train_loader:
       print(f"Views: {len(views)}, Shape: {views[0].shape}")
       print(f"Min: {views[0].min()}, Max: {views[0].max()}")
       break
   ```

4. **Check for data issues**:
   - Corrupted images
   - Wrong normalization values
   - Empty batches

---

### Loss is Negative

**This should not happen** with this implementation. The original notebook had a bug that caused negative loss - it's fixed here.

If you see negative loss:

1. Verify you're using the correct loss class:
   ```python
   from dino.loss import DinoLoss  # Use this
   ```

2. Check that outputs are correctly chunked:
   ```python
   # Correct: 8 student views, 2 teacher views
   print(f"Student output shape: {student_output.shape}")
   print(f"Teacher output shape: {teacher_output.shape}")
   ```

---

### Out of Memory (OOM)

**Symptoms**: `CUDA out of memory` error

**Solutions**:

1. **Reduce batch size**:
   ```yaml
   data:
     batch_size: 16  # or 8
   ```

2. **Reduce local crops**:
   ```yaml
   augmentation:
     num_local_views: 4  # Instead of 6
   ```

3. **Use smaller backbone**:
   ```yaml
   model:
     backbone: resnet18  # Instead of resnet50
   ```

4. **Enable gradient checkpointing** (advanced):
   ```python
   from torch.utils.checkpoint import checkpoint
   # Wrap forward passes
   ```

5. **Use mixed precision**:
   ```python
   from torch.cuda.amp import autocast
   with autocast():
       outputs = model(inputs)
   ```

---

### Out of Memory with ViT Backbone

ViT backbones require more memory than ResNet.

**Solutions**:

1. **Reduce batch size significantly**:
   ```yaml
   data:
     batch_size: 8  # ViT needs smaller batches
   ```

2. **Use smaller ViT variant**:
   ```yaml
   model:
     backbone: dino_vits16  # Instead of dino_vitb16
   ```

   Memory requirements:
   - `dino_vits8/vits16`: ~8GB with batch_size=16
   - `dino_vitb8/vitb16`: ~16GB with batch_size=16

---

### Slow Training

**Symptoms**: Training is unexpectedly slow, GPU utilization low

**Solutions**:

1. **Increase data loading workers**:
   ```yaml
   data:
     num_workers: 8  # Match CPU cores
   ```

2. **Enable pin memory**:
   ```yaml
   data:
     pin_memory: true
   ```

3. **Check GPU utilization**:
   ```bash
   nvidia-smi -l 1
   ```
   - If GPU util < 50%: Data loading bottleneck
   - If GPU util > 90%: Normal

4. **Use SSD for data**: Spinning disks are slow for random access

---

## Data Issues

### Dataset Not Found

**Error**: `FileNotFoundError` or `Dataset not found`

**Solutions**:

1. **Check data path**:
   ```yaml
   data:
     data_path: ./data  # Verify this path exists
   ```

2. **For ImageNette**: It should auto-download. Check internet connection.

3. **For ImageNet100**: Manual download required:
   ```bash
   kaggle datasets download -d ambityga/imagenet100
   unzip imagenet100.zip -d ./data/imagenet100
   ```

---

### Corrupted Images

**Symptoms**: `PIL.UnidentifiedImageError` or similar

**Solutions**:

1. **Find corrupted files**:
   ```python
   from PIL import Image
   import os

   for root, dirs, files in os.walk('./data'):
       for f in files:
           if f.endswith(('.jpg', '.png', '.jpeg')):
               try:
                   Image.open(os.path.join(root, f)).verify()
               except:
                   print(f"Corrupted: {os.path.join(root, f)}")
   ```

2. **Remove or replace corrupted files**

---

## Checkpoint Issues

### Cannot Load Checkpoint

**Error**: `KeyError` or shape mismatch when loading

**Causes**:

1. **Architecture mismatch**: Checkpoint from different model configuration
2. **Missing keys**: Checkpoint from older code version

**Solutions**:

1. **Use strict=False** (may lose some weights):
   ```python
   model.load_state_dict(checkpoint['state_dict'], strict=False)
   ```

2. **Verify config matches**:
   ```python
   checkpoint = torch.load('checkpoint.pth')
   print(checkpoint['config'])  # Compare with current config
   ```

3. **Retrain from scratch** if architecture changed significantly

---

### Checkpoint Too Large

**Symptoms**: Checkpoint files are very large (>1GB)

**Cause**: Saving unnecessary data (optimizer state, etc.)

**Solution**: For inference, save only model weights:
```python
torch.save({
    'model_state_dict': model.state_dict(),
}, 'model_only.pth')
```

---

## Configuration Issues

### YAML Parsing Errors

**Error**: `yaml.scanner.ScannerError`

**Common causes**:

1. **Indentation errors**: YAML requires consistent indentation
2. **Special characters**: Escape colons in strings

**Solutions**:
```yaml
# Wrong
data:
dataset: imagenette  # Missing indentation

# Correct
data:
  dataset: imagenette
```

---

### Type Errors in Config

**Error**: `TypeError` when loading config

**Solution**: Check types match dataclass definitions:
```yaml
# Wrong
data:
  batch_size: "32"  # String, should be int

# Correct
data:
  batch_size: 32
```

---

## GPU Issues

### CUDA Not Available

```python
import torch
print(torch.cuda.is_available())  # Should be True
```

**Solutions**:

1. **Reinstall PyTorch with CUDA**: [pytorch.org](https://pytorch.org/get-started/locally/)
2. **Check NVIDIA drivers**: `nvidia-smi`
3. **Verify CUDA version compatibility**

---

### Wrong GPU Selected

**Solution**: Set GPU explicitly:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py
```

Or in Python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

---

## Known Issues

### Negative Loss Values (Historical Bug - FIXED)

**Symptom**: Loss values are negative and increasing in magnitude

**Root cause**: The original notebook implementation had a bug where it iterated over raw output tensors instead of chunked probability distributions.

**Wrong (original)**:
```python
for i, teacher_output in enumerate(teacher_outputs):  # Raw tensors!
    for j, student_output in enumerate(student_outputs):
        loss = -torch.sum(teacher_output * student_output)
```

**Correct (fixed)**:
```python
# Chunk outputs into views first
student_views = student_log_probs.chunk(ncrops)
teacher_views = teacher_probs.chunk(n_global_crops)

for i, teacher_prob in enumerate(teacher_views):  # Chunked distributions
    for j, student_log_prob in enumerate(student_views):
        loss = -torch.sum(teacher_prob * student_log_prob)
```

This implementation includes the fix - loss values should always be positive. If you see negative loss, verify you're using the correct `DinoLoss` class from this codebase.

---

### YAML Tuple Serialization

**Symptom**: Tuples in config become lists after saving and loading YAML

**Example**:
```python
# Before save: normalize_mean is tuple (0.485, 0.456, 0.406)
config.save_yaml('config.yaml')
config2 = DinoConfig.from_yaml('config.yaml')
# After load: normalize_mean is list [0.485, 0.456, 0.406]
```

**Why this happens**: YAML has no native tuple type. All sequences are represented as lists.

**Workaround**: This is handled internally. The dataclass accepts both tuples and lists for sequence fields. If you need a tuple specifically, convert after loading:

```python
mean = tuple(config.augmentation.normalize_mean)
```

---

### Checkpoint Compatibility

**Issue**: Checkpoints from different code versions may not load

**Symptoms**:
- `KeyError` when loading checkpoint
- Shape mismatch errors
- Missing keys in state dict

**Cause**: The codebase was significantly refactored. Old checkpoints from the notebook version use different key names and structures.

**Solution**: Retrain from scratch rather than trying to maintain backward compatibility. The code is designed for forward compatibility - checkpoints saved with this version will continue to work.

**If you must use old checkpoints**:
```python
# Load with strict=False (may lose some weights)
checkpoint = torch.load('old_checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'], strict=False)
# Warning: Some weights may be missing or mismatched
```

---

## Weights & Biases Issues

### W&B Login Required

**Error**: `wandb: ERROR You must be logged in to use wandb`

**Solution**:
```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### W&B Run Not Resuming

**Symptom**: New run created instead of resuming

**Solution**: Ensure checkpoint contains `wandb_run_id`:
```python
checkpoint = torch.load('checkpoint.pth')
print(checkpoint.get('wandb_run_id'))  # Should not be None
```

If resuming manually:
```python
wandb.init(
    id=checkpoint['wandb_run_id'],
    resume="must"
)
```

---

## Kaggle-Specific Issues

### Cannot Access Dataset

**Error**: Dataset not found at `/kaggle/input/`

**Solutions**:

1. **Check dataset is attached**: In Kaggle notebook, go to "Add Data" and attach the dataset
2. **Verify path**: Use the correct Kaggle input path
   ```python
   import os
   print(os.listdir('/kaggle/input/'))
   ```

### Output Quota Exceeded

**Error**: `OSError: [Errno 28] No space left on device`

**Solution**: Save checkpoints less frequently:
```yaml
checkpoint:
  save_every_n_epochs: 20  # Instead of every epoch
```

### W&B on Kaggle

**Setup**:
```python
import wandb
from kaggle_secrets import UserSecretsClient

# Get secret from Kaggle
secrets = UserSecretsClient()
wandb_api_key = secrets.get_secret("wandb_api_key")

# Login
wandb.login(key=wandb_api_key)
```

---

## Getting Help

If your issue isn't listed here:

1. **Check the logs**: Look for error messages in `./logs/`
2. **Enable debug mode**: Set `log_verbosity: debug` in config
3. **Minimal reproduction**: Create a minimal example that reproduces the issue
4. **Open an issue**: Include error message, config, and environment info

---

## See Also

- [Installation](getting-started/installation.md) - Setup guide
- [Configuration](guides/configuration.md) - Config reference
- [Performance](advanced/performance.md) - Optimization tips
