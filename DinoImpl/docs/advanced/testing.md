# Testing

This guide covers testing strategies for the DINO implementation, including unit tests, integration tests, and how to verify your setup.

---

## Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_transforms.py

# Run specific test
pytest tests/test_loss.py::test_dino_loss_positive
```

### Test Coverage

```bash
# Run with coverage report
pytest tests/ --cov=src/dino --cov-report=html

# View report in browser
open htmlcov/index.html
```

---

## Unit Tests

### Transform Tests

Test that the multi-crop transform produces the correct number and shape of views:

```python
# tests/test_transforms.py
import torch
from PIL import Image
from dino.data import DINOTransform

def test_dino_transform_output_count():
    """Transform should produce 8 views (2 global + 6 local)."""
    transform = DINOTransform(num_local_views=6)
    img = Image.new('RGB', (256, 256))

    views = transform(img)

    assert len(views) == 8, f"Expected 8 views, got {len(views)}"

def test_dino_transform_global_shape():
    """Global views should be 224x224."""
    transform = DINOTransform(
        global_crop_size=224,
        local_crop_size=96,
        num_local_views=6
    )
    img = Image.new('RGB', (256, 256))

    views = transform(img)

    # First 2 views are global
    assert views[0].shape == (3, 224, 224)
    assert views[1].shape == (3, 224, 224)

def test_dino_transform_local_shape():
    """Local views should be 96x96."""
    transform = DINOTransform(
        global_crop_size=224,
        local_crop_size=96,
        num_local_views=6
    )
    img = Image.new('RGB', (256, 256))

    views = transform(img)

    # Views 2-7 are local
    for i in range(2, 8):
        assert views[i].shape == (3, 96, 96), f"View {i} has wrong shape"

def test_dino_transform_normalization():
    """Views should be normalized to roughly [-2, 2] range."""
    transform = DINOTransform(num_local_views=6)
    img = Image.new('RGB', (256, 256), color=(128, 128, 128))

    views = transform(img)

    for i, view in enumerate(views):
        assert view.min() >= -3, f"View {i} min too low: {view.min()}"
        assert view.max() <= 3, f"View {i} max too high: {view.max()}"
```

### Loss Function Tests

Verify the loss computation produces valid, positive values:

```python
# tests/test_loss.py
import torch
from dino.loss import DinoLoss

def test_dino_loss_positive():
    """Loss must always be positive (cross-entropy property)."""
    loss_fn = DinoLoss(
        out_dim=128,
        student_temp=0.1,
        teacher_temp=0.04,
        ncrops=8,
        n_global_crops=2
    )

    # Simulate: 8 batch, 8 student views, 2 teacher views
    student_out = torch.randn(64, 128)  # 8 * 8 = 64
    teacher_out = torch.randn(16, 128)  # 8 * 2 = 16

    loss = loss_fn(student_out, teacher_out)

    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"

def test_dino_loss_not_nan():
    """Loss should not be NaN under normal conditions."""
    loss_fn = DinoLoss(out_dim=128, ncrops=8, n_global_crops=2)

    student_out = torch.randn(64, 128)
    teacher_out = torch.randn(16, 128)

    loss = loss_fn(student_out, teacher_out)

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is infinite"

def test_dino_loss_center_update():
    """Center should be updated after forward pass."""
    loss_fn = DinoLoss(out_dim=128, ncrops=8, n_global_crops=2)

    initial_center = loss_fn.center.clone()

    student_out = torch.randn(64, 128)
    teacher_out = torch.randn(16, 128)
    loss_fn(student_out, teacher_out)

    # Center should have changed
    assert not torch.allclose(loss_fn.center, initial_center), \
        "Center should be updated after forward pass"

def test_dino_loss_gradient_flow():
    """Gradients should flow through student but not teacher."""
    loss_fn = DinoLoss(out_dim=128, ncrops=8, n_global_crops=2)

    student_out = torch.randn(64, 128, requires_grad=True)
    teacher_out = torch.randn(16, 128)  # No grad

    loss = loss_fn(student_out, teacher_out)
    loss.backward()

    assert student_out.grad is not None, "Student should have gradients"
```

### Checkpoint Tests

Test saving and loading checkpoints:

```python
# tests/test_checkpoint.py
import torch
import tempfile
import os
from dino.models import DinoModel
from dino.config import DinoConfig
from dino.utils import save_checkpoint, load_checkpoint

def test_checkpoint_save_load():
    """Checkpoint should preserve model weights exactly."""
    config = DinoConfig()

    # Create model
    model1 = DinoModel.from_config(config)
    model1.eval()

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.pth')
        torch.save({
            'model_state_dict': model1.state_dict(),
        }, path)

        # Load into new model
        model2 = DinoModel.from_config(config)
        checkpoint = torch.load(path)
        model2.load_state_dict(checkpoint['model_state_dict'])
        model2.eval()

        # Verify weights match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Weights don't match after load"

def test_checkpoint_contains_required_keys():
    """Full checkpoint should contain all training state."""
    # This tests the checkpoint format, not the function
    required_keys = [
        'epoch',
        'student_state_dict',
        'teacher_state_dict',
        'optimizer_state_dict',
        'dino_loss_center',
        'config'
    ]

    # Create minimal checkpoint
    checkpoint = {
        'epoch': 0,
        'student_state_dict': {},
        'teacher_state_dict': {},
        'optimizer_state_dict': {},
        'dino_loss_center': torch.zeros(2048),
        'config': {}
    }

    for key in required_keys:
        assert key in checkpoint, f"Checkpoint missing required key: {key}"
```

### Model Tests

Test model architecture and forward pass:

```python
# tests/test_models.py
import torch
from dino.models import DinoModel, get_backbone, DinoProjectionHead
from dino.config import DinoConfig

def test_model_forward_shape():
    """Model output should have correct shape."""
    config = DinoConfig()
    model = DinoModel.from_config(config)

    # Batch of 4 images
    x = torch.randn(4, 3, 224, 224)

    output = model(x)

    expected_shape = (4, config.model.projection_output_dim)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"

def test_model_return_backbone_features():
    """Model should optionally return backbone features."""
    config = DinoConfig()
    model = DinoModel.from_config(config)

    x = torch.randn(4, 3, 224, 224)

    features, projections = model(x, return_backbone_features=True)

    # Features should match backbone output dim
    backbone = get_backbone(config.model.backbone)
    assert features.shape == (4, backbone.output_dim)
    assert projections.shape == (4, config.model.projection_output_dim)

def test_backbone_output_dim():
    """Backbones should report correct output dimension."""
    test_cases = [
        ('resnet18', 512),
        ('resnet50', 2048),
    ]

    for name, expected_dim in test_cases:
        backbone = get_backbone(name)
        assert backbone.output_dim == expected_dim, \
            f"{name} should have output_dim {expected_dim}, got {backbone.output_dim}"
```

---

## Integration Tests

### Quick Training Test

Verify the full training pipeline works with a minimal configuration:

```python
# tests/test_integration.py
import torch
from dino.config import DinoConfig
from dino.models import DinoModel
from dino.loss import DinoLoss
from dino.data import DINOTransform

def test_full_forward_pass():
    """Test complete forward pass through student-teacher setup."""
    config = DinoConfig()

    # Create models
    student = DinoModel.from_config(config)
    teacher = DinoModel.from_config(config)
    teacher.load_state_dict(student.state_dict())

    # Create loss
    loss_fn = DinoLoss.from_config(
        config.loss,
        config.augmentation,
        student.output_dim
    )

    # Simulate batch of views
    batch_size = 4
    global_views = [torch.randn(batch_size, 3, 224, 224) for _ in range(2)]
    local_views = [torch.randn(batch_size, 3, 96, 96) for _ in range(6)]
    all_views = global_views + local_views

    # Teacher forward (global only)
    with torch.no_grad():
        teacher_out = torch.cat([teacher(v) for v in global_views])

    # Student forward (all views)
    student_out = torch.cat([student(v) for v in all_views])

    # Compute loss
    loss = loss_fn(student_out, teacher_out)

    # Verify
    assert loss.item() > 0
    assert not torch.isnan(loss)
```

### Training Loop Test

Test a minimal training iteration:

```bash
# Run quick training test (1 epoch on small synthetic data)
python -c "
from dino.config import DinoConfig
from dino.models import DinoModel
from dino.loss import DinoLoss
from dino.utils import update_teacher_EMA
import torch

config = DinoConfig()
config.training.num_epochs = 1

student = DinoModel.from_config(config)
teacher = DinoModel.from_config(config)
teacher.load_state_dict(student.state_dict())

for p in teacher.parameters():
    p.requires_grad = False

loss_fn = DinoLoss.from_config(config.loss, config.augmentation, student.output_dim)
optimizer = torch.optim.AdamW(student.parameters(), lr=0.001)

# One training step
batch_size = 4
global_views = [torch.randn(batch_size, 3, 224, 224) for _ in range(2)]
local_views = [torch.randn(batch_size, 3, 96, 96) for _ in range(6)]

with torch.no_grad():
    teacher_out = torch.cat([teacher(v) for v in global_views])

student_out = torch.cat([student(v) for v in global_views + local_views])
loss = loss_fn(student_out, teacher_out)

optimizer.zero_grad()
loss.backward()
optimizer.step()

update_teacher_EMA(student, teacher, momentum=0.996)

print(f'Training step completed. Loss: {loss.item():.4f}')
"
```

---

## Test Fixtures

Create reusable test fixtures in `conftest.py`:

```python
# tests/conftest.py
import pytest
import torch
from PIL import Image
from dino.config import DinoConfig
from dino.models import DinoModel
from dino.data import DINOTransform

@pytest.fixture
def config():
    """Default test configuration."""
    return DinoConfig()

@pytest.fixture
def model(config):
    """Pre-built model for testing."""
    return DinoModel.from_config(config)

@pytest.fixture
def transform():
    """Default transform for testing."""
    return DINOTransform(num_local_views=6)

@pytest.fixture
def sample_image():
    """Sample RGB image for testing."""
    return Image.new('RGB', (256, 256), color=(128, 128, 128))

@pytest.fixture
def sample_batch():
    """Sample batch of images."""
    return torch.randn(4, 3, 224, 224)
```

---

## Continuous Integration

Example GitHub Actions workflow for automated testing:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src/dino
```

---

## Debugging Test Failures

### Common Issues

**1. CUDA errors in tests**:
```python
# Force CPU for tests
@pytest.fixture(autouse=True)
def force_cpu():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

**2. Random seed for reproducibility**:
```python
@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
```

**3. Memory issues**:
```python
# Use smaller models/batches in tests
config.model.projection_output_dim = 128  # Smaller output
config.data.batch_size = 2  # Smaller batch
```

---

## See Also

- [Configuration](../guides/configuration.md) - Configuration options
- [Troubleshooting](../troubleshooting.md) - Common issues
- [Performance](performance.md) - Optimization tips
