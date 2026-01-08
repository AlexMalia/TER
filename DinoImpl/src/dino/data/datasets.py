"""Dataset wrappers and utilities."""

import torch
from torch.utils.data import Dataset, random_split
import torchvision
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def get_dataset(
    dataset_name: str,
    data_path: str,
    transform: Optional[Callable] = None,
    download: bool = True,
    train: bool = True
) -> Dataset:
    """
    Get dataset by name.

    Supported datasets:
    - cifar100: CIFAR-100
    - imagenette: ImageNette (subset of ImageNet)

    Args:
        dataset_name: Name of the dataset
        data_path: Path to data directory
        transform: Transform to apply to images
        download: Whether to download the dataset if not present
        train: Whether to load training or test split

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset_name is not supported

    Example:
        >>> from dino.data.transforms import DINOTransform
        >>> transform = DINOTransform()
        >>> dataset = get_dataset('cifar100', './data', transform=transform)
        >>> print(len(dataset))
        50000
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'cifar100':
        return torchvision.datasets.CIFAR100(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )
    elif dataset_name == 'imagenette':
        # Imagenette uses 'train' or 'val' as split argument
        split = 'train' if train else 'val'
        return torchvision.datasets.Imagenette(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: cifar100, imagenette"
        )


def create_train_val_test_splits(
    dataset: Dataset,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: Full dataset
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Example:
        >>> dataset = get_dataset('cifar100', './data')
        >>> train_ds, val_ds, test_ds = create_train_val_test_splits(dataset)
        >>> print(len(train_ds), len(val_ds), len(test_ds))
        35000 7500 7500
    """
    if train_split + val_split > 1.0:
        raise ValueError(
            f"train_split ({train_split}) + val_split ({val_split}) "
            f"must be <= 1.0"
        )

    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    logger.info(
        f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}"
    )

    # Create generator for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


class MultiCropDataset(Dataset):
    """
    Wrapper dataset that applies multi-crop transformations.

    This is useful when you want to use a dataset that doesn't natively
    support multi-crop transformations.

    Args:
        base_dataset: Base dataset
        transform: Multi-crop transform (e.g., DINOTransform)

    Example:
        >>> from torchvision.datasets import CIFAR100
        >>> from dino.data.transforms import DINOTransform
        >>> base_dataset = CIFAR100(root='./data', train=True, download=True)
        >>> transform = DINOTransform()
        >>> dataset = MultiCropDataset(base_dataset, transform)
        >>> views, label = dataset[0]
        >>> len(views)  # 2 global + 6 local
        8
    """

    def __init__(self, base_dataset: Dataset, transform: Callable):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get item with multi-crop transformations.

        Returns:
            Tuple of (views, label) where views is a list of transformed images
        """
        img, label = self.base_dataset[idx]
        views = self.transform(img)
        return views, label
