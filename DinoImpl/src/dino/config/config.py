"""Configuration dataclasses for DINO training."""

from dataclasses import dataclass, field, asdict
import argparse
from pathlib import Path
from typing import  Tuple
import yaml


@dataclass
class DataConfig:
    """Data loading configuration."""

    dataset: str
    data_path: str
    batch_size: int
    num_workers: int
    do_pin_memory: bool
    train_split: float
    val_split: float
    seed: int


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # Global
    num_global_views: int
    global_crop_size: int
    global_crop_scale_min: float
    global_crop_scale_max: float

    # Local
    num_local_views: int
    local_crop_size: int
    local_crop_scale_min: float
    local_crop_scale_max: float

    # Color jitter
    color_jitter_prob: float
    color_jitter_brightness: float
    color_jitter_contrast: float
    color_jitter_saturation: float
    color_jitter_hue: float

    # Other augmentations
    horizontal_flip_prob: float
    grayscale_prob: float
    gaussian_blur_sigma_min: float
    gaussian_blur_sigma_max: float
    solarization_prob: float
    solarization_threshold: int

    # Normalization
    normalize_mean: Tuple[float, float, float]
    normalize_std: Tuple[float, float, float]

    @property
    def num_total_views(self) -> int:
        """Total number of crops (global + local)."""
        return self.num_global_views + self.num_local_views


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    backbone: str
    is_backbone_pretrained: bool
    projection_hidden_dim: int
    projection_bottleneck_dim: int
    projection_output_dim: int
    use_weight_norm: bool


@dataclass
class LossConfig:
    """Loss function configuration."""

    student_temp: float
    teacher_temp: float
    center_momentum: float


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    optimizer: str
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    scheduler: str
    warmup_epochs: int
    min_lr: float
    warmup_start_lr: float


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_epochs: int
    teacher_momentum: float
    teacher_momentum_schedule: bool
    teacher_momentum_final: float
    gradient_clip: float
    gradient_accumulation_steps: int
    mixed_precision: bool
    seed: int
    device: str


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    checkpoint_dir: str
    save_every_n_epochs: int
    keep_last_n: int
    save_best: bool
    resume_from: str


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str
    log_every_n_iters: int
    log_verbosity: str
    use_wandb: bool
    wandb_project: str
    wandb_entity: str
    wandb_run_name: str


_CONFIG_CLASSES = {
    'data_config': DataConfig,
    'augmentation_config': AugmentationConfig,
    'model_config': ModelConfig,
    'loss_config': LossConfig,
    'optimizer_config': OptimizerConfig,
    'scheduler_config': SchedulerConfig,
    'training_config': TrainingConfig,
    'checkpoint_config': CheckpointConfig,
    'logging_config': LoggingConfig,
}

_ARG_MAPPING = {
    'dataset': [('data_config', 'dataset')],
    'data_path': [('data_config', 'data_path')],
    'batch_size': [('data_config', 'batch_size')],
    'num_workers': [('data_config', 'num_workers')],
    'backbone': [('model_config', 'backbone')],
    'output_dim': [('model_config', 'projection_output_dim')],
    'epochs': [('training_config', 'num_epochs')],
    'lr': [('optimizer_config', 'lr')],
    'checkpoint_dir': [('checkpoint_config', 'checkpoint_dir')],
    'log_dir': [('logging_config', 'log_dir')],
    'log_verbosity': [('logging_config', 'log_verbosity')],
    'resume': [('checkpoint_config', 'resume_from')],
    'device': [('training_config', 'device')],
    'seed': [('training_config', 'seed'), ('data_config', 'seed')], 
}


@dataclass
class DinoConfig:
    """Main DINO configuration."""

    data_config: DataConfig
    augmentation_config: AugmentationConfig
    model_config: ModelConfig
    loss_config: LossConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    training_config: TrainingConfig
    checkpoint_config: CheckpointConfig
    logging_config: LoggingConfig

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
    def from_yaml_and_args(cls, yaml_path: str, args: argparse.Namespace) -> 'DinoConfig':
        """Load configuration from YAML file and override with command-line arguments."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        print(f"Loaded configuration from {yaml_path}:")
        print(data)

        config_dict = {}
        for key, value in data.items():
            if key in _CONFIG_CLASSES and isinstance(value, dict):
                config_dict[key] = _CONFIG_CLASSES[key](**value)
                
        config = cls(**config_dict)

        for arg_name, targets in _ARG_MAPPING.items():
            arg_value = getattr(args, arg_name, None)
            if arg_value is not None:
                for section, field in targets:
                    setattr(getattr(config, section), field, arg_value)
    
        return config


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
