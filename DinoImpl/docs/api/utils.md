# Utils API Reference

API documentation for utility functions.

---

## EMA

### update_teacher_EMA

::: dino.utils.update_teacher_EMA
    options:
      show_root_heading: true
      show_source: true

### get_momentum_schedule

::: dino.utils.get_momentum_schedule
    options:
      show_root_heading: true
      show_source: true

---

## Checkpointing

### save_checkpoint

::: dino.utils.save_checkpoint
    options:
      show_root_heading: true
      show_source: true

### load_checkpoint

::: dino.utils.load_checkpoint
    options:
      show_root_heading: true
      show_source: true

---

## History

::: dino.utils.History
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - record_iteration
        - record_epoch
        - get_metric
        - get_iterations
        - get_epochs
        - save
        - load
        - plot_loss
        - plot_learning_rate
        - plot_momentum
        - plot_all
        - to_dataframe

---

## Usage Examples

### EMA Updates

```python
from dino.utils import update_teacher_EMA, get_momentum_schedule

# Simple EMA update
update_teacher_EMA(student, teacher, momentum=0.996)

# With momentum schedule
schedule = get_momentum_schedule(
    base=0.996,
    final=1.0,
    num_epochs=100,
    niter_per_epoch=len(train_loader)
)

for iteration in range(total_iterations):
    # ... training step ...
    momentum = schedule[iteration]
    update_teacher_EMA(student, teacher, momentum)
```

### Saving Checkpoints

```python
from dino.utils import save_checkpoint

save_checkpoint(
    path='checkpoints/checkpoint_epoch_50.pth',
    epoch=50,
    iteration=10000,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    loss_fn=loss_fn,
    config=config,
    metrics={'loss': 2.5, 'knn_acc': 0.75}
)
```

### Loading Checkpoints

```python
from dino.utils import load_checkpoint

checkpoint = load_checkpoint(
    path='checkpoints/checkpoint_latest.pth',
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    loss_fn=loss_fn
)

# Resume from checkpoint
start_epoch = checkpoint['epoch'] + 1
iteration = checkpoint['iteration']
```

### Training History

```python
from dino.utils import History

# Create history tracker
history = History(metadata={'config': config.to_dict()})

# Record during training
for epoch in range(num_epochs):
    for iteration, batch in enumerate(train_loader):
        # ... training ...
        history.record_iteration(total_iter, {
            'loss': loss.item(),
            'learning_rate': current_lr,
            'momentum': current_momentum
        })
        total_iter += 1

    history.record_epoch(epoch, {
        'loss': epoch_loss,
        'learning_rate': current_lr,
        'momentum': current_momentum
    })

# Save history
history.save('training_history.json')

# Load history
history = History.load('training_history.json')

# Plot metrics
history.plot_loss(level='epoch')
history.plot_learning_rate(level='iteration')
history.plot_all(level='epoch', save_path='training_plots.png')

# Export to DataFrame
df = history.to_dataframe(level='epoch')
print(df.head())
```

### Getting Metrics

```python
# Get specific metric
losses = history.get_metric('loss', level='epoch')
learning_rates = history.get_metric('learning_rate', level='iteration')

# Get indices
iterations = history.get_iterations()
epochs = history.get_epochs()
```
