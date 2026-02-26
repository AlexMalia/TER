"""Data augmentation transforms for DINO."""

import torch
from torchvision import transforms
from typing import List, Tuple
from dino.config import AugmentationConfig

class DINOTransform:
    """
    Multi-crop data augmentation strategy for DINO.

    Creates multiple augmented views of the same image:
    - 2 global views at higher resolution
    - N local views at lower resolution

    This encourages the model to learn global and local features.

    Args:
        num_local_views: Number of local crops
        global_crop_size: Size of global crops
        local_crop_size: Size of local crops
        global_crop_scale: Tuple of (min_scale, max_scale) for global crops
        local_crop_scale: Tuple of (min_scale, max_scale) for local crops
        color_jitter_prob: Probability of applying color jitter
        color_jitter_params: Tuple of (brightness, contrast, saturation, hue)
        horizontal_flip_prob: Probability of horizontal flip
        grayscale_prob: Probability of converting to grayscale
        gaussian_blur_sigma: Tuple of (min_sigma, max_sigma) for Gaussian blur
        solarization_prob: Probability of applying solarization (only to 2nd global view)
        solarization_threshold: Threshold for solarization
        normalize_mean: Mean for normalization
        normalize_std: Standard deviation for normalization
    """

    def __init__(
        self,
        num_local_views: int = 6,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
        color_jitter_prob: float = 0.8,
        color_jitter_params: Tuple[float, float, float, float] = (0.4, 0.4, 0.2, 0.1),
        horizontal_flip_prob: float = 0.5,
        grayscale_prob: float = 0.2,
        gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0),
        solarization_prob: float = 0.2,
        solarization_threshold: int = 128,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.num_local_views = num_local_views

        # Common augmentations for all crops
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
            transforms.RandomApply(
                [transforms.ColorJitter(*color_jitter_params)],
                p=color_jitter_prob
            ),
            transforms.RandomGrayscale(p=grayscale_prob),
        ])

        # Normalization
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])

        # Global view 1 (no solarization)
        self.global_t1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crop_scale),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=23, sigma=gaussian_blur_sigma),
            normalize,
        ])

        # Global view 2 (with solarization)
        self.global_t2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crop_scale),
            flip_and_color_jitter,
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=gaussian_blur_sigma)],
                p=0.1
            ),
            transforms.RandomSolarize(threshold=solarization_threshold, p=solarization_prob),
            normalize,
        ])

        # Local views
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crop_scale),
            flip_and_color_jitter,
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=gaussian_blur_sigma)],
                p=0.5
            ),
            normalize,
        ])

    def __call__(self, img) -> List[torch.Tensor]:
        """
        Apply transformations to create multiple views.

        Args:
            img: PIL Image

        Returns:
            List of augmented views (2 global + num_local_views local)
        """
        views = []

        # Add 2 global views
        views.append(self.global_t1(img))
        views.append(self.global_t2(img))

        # Add local views
        for _ in range(self.num_local_views):
            views.append(self.local_transform(img))

        return views

    @classmethod
    def from_config(cls, augmentation: AugmentationConfig) -> 'DINOTransform':
        """
        Factory method to create DINOTransform from AugmentationConfig.

        Args:
            augmentation: AugmentationConfig instance

        Returns:
            DINOTransform instance
        """
        return cls(
            num_local_views=augmentation.num_local_views,
            global_crop_size=augmentation.global_crop_size,
            local_crop_size=augmentation.local_crop_size,
            global_crop_scale=(
                augmentation.global_crop_scale_min,
                augmentation.global_crop_scale_max
            ),
            local_crop_scale=(
                augmentation.local_crop_scale_min,
                augmentation.local_crop_scale_max
            ),
            color_jitter_prob=augmentation.color_jitter_prob,
            color_jitter_params=(
                augmentation.color_jitter_brightness,
                augmentation.color_jitter_contrast,
                augmentation.color_jitter_saturation,
                augmentation.color_jitter_hue
            ),
            horizontal_flip_prob=augmentation.horizontal_flip_prob,
            grayscale_prob=augmentation.grayscale_prob,
            gaussian_blur_sigma=(
                augmentation.gaussian_blur_sigma_min,
                augmentation.gaussian_blur_sigma_max
            ),
            solarization_prob=augmentation.solarization_prob,
            solarization_threshold=augmentation.solarization_threshold,
            normalize_mean=augmentation.normalize_mean,
            normalize_std=augmentation.normalize_std,
        )
