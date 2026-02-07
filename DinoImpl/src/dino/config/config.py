"""Configuration dataclasses for DINO training."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple
import yaml


@dataclass
class DataConfig:
    """Data loading configuration."""

    dataset: str = "imagenette"
    data_path: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.7
    val_split: float = 0.15
    seed: int = 42


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""

    n_global_crops: int = 2
    num_local_views: int = 6
    global_crop_size: int = 224
    local_crop_size: int = 96
    global_crop_scale_min: float = 0.4
    global_crop_scale_max: float = 1.0
    local_crop_scale_min: float = 0.05
    local_crop_scale_max: float = 0.4

    # Color jitter
    color_jitter_prob: float = 0.8
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1

    # Other augmentations
    horizontal_flip_prob: float = 0.5
    grayscale_prob: float = 0.2
    gaussian_blur_sigma_min: float = 0.1
    gaussian_blur_sigma_max: float = 2.0
    solarization_prob: float = 0.2
    solarization_threshold: int = 128

    # Normalization
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def ncrops(self) -> int:
        """Total number of crops (global + local)."""
        return self.n_global_crops + self.num_local_views


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    backbone: str = "resnet18"
    backbone_pretrained: bool = False
    projection_hidden_dim: int = 1024
    projection_bottleneck_dim: int = 256
    projection_output_dim: int = 2048
    use_weight_norm: bool = True


@dataclass
class LossConfig:
    """Loss function configuration."""

    student_temp: float = 0.1
    teacher_temp: float = 0.04
    center_momentum: float = 0.9


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    optimizer: str = "adamw"
    lr: float = 0.001
    weight_decay: float = 0.04
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    warmup_start_lr: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_epochs: int = 100
    teacher_momentum: float = 0.996
    teacher_momentum_schedule: bool = True
    teacher_momentum_final: float = 1.0
    gradient_clip: Optional[float] = 3.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    seed: int = 42
    device: str = "cuda"


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 1
    save_every_n_iters: Optional[int] = None
    keep_last_n: int = 5
    save_best: bool = True
    resume_from: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = "./logs"
    log_every_n_iters: int = 10
    log_verbosity: str = "info"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None


_CONFIG_CLASSES = {
    'data': DataConfig,
    'augmentation': AugmentationConfig,
    'model': ModelConfig,
    'loss': LossConfig,
    'optimizer': OptimizerConfig,
    'scheduler': SchedulerConfig,
    'training': TrainingConfig,
    'checkpoint': CheckpointConfig,
    'logging': LoggingConfig,
}


@dataclass
class DinoConfig:
    """Main DINO configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self):
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @classmethod
    def from_yaml(cls, path: str) -> 'DinoConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config_dict = {}
        for key, value in data.items():
            if key in _CONFIG_CLASSES and isinstance(value, dict):
                config_dict[key] = _CONFIG_CLASSES[key](**value)

        return cls(**config_dict)

    def __str__(self) -> str:
        """String representation of configuration."""
        lines = ["DinoConfig:"]
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name)
            lines.append(f"  {field_name}:")
            for sub_field in field_value.__dataclass_fields__:
                sub_value = getattr(field_value, sub_field)
                lines.append(f"    {sub_field}: {sub_value}")
        return "\n".join(lines)
