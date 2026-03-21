"""Dataset wrappers and utilities."""

import os
import glob
import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
import torchvision
from torchvision.datasets import ImageFolder
from typing import Tuple, Optional, Callable, List
import logging

from .custom_datasets import AMRGraphDataset, AMRGraphDatasetGlobal

logger = logging.getLogger(__name__)


def get_dataset(
    dataset_name: str,
    data_path: str,
    transform: Optional[Callable] = None,
    download: bool = True,
    train: bool = True,
    bfs_depth: int = 1
) -> Dataset:
    """
    Get dataset by name.

    Supported datasets:
    - imagenette: ImageNette (subset of ImageNet)
    - imagenet100: ImageNet100 (100-class subset of ImageNet from Kaggle)

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
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'imagenette':
        # Imagenette uses 'train' or 'val' as split argument
        split = 'train' if train else 'val'
        imagenette = torchvision.datasets.Imagenette(
            root=data_path,
            split=split,
            download=download,
            transform=transform,
            size='320px'
        )
        logger.info(f"Loaded Imagenette dataset ({split}) with {len(imagenette)} samples")
        logger.info(f"Image size example: {imagenette[0][0][0].shape} (C, H, W)")
        return imagenette
    
    elif dataset_name == 'imagenet100':
        # ImageNet100 from Kaggle: https://www.kaggle.com/datasets/ambityga/imagenet100
        # Structure: data_path/train.X1/, train.X2/, train.X3/, train.X4/ and data_path/val.X/
        if train:
            # Find all train.X* folders and combine them
            train_patterns = sorted(glob.glob(os.path.join(data_path, 'train.X*')))
            if not train_patterns:
                raise FileNotFoundError(
                    f"ImageNet100 train folders not found at: {data_path}\n"
                    f"Expected folders like train.X1, train.X2, etc.\n"
                    f"Please download the dataset from Kaggle:\n"
                    f"  kaggle datasets download -d ambityga/imagenet100\n"
                    f"Then extract it to: {data_path}"
                )

            logger.info(f"Loading ImageNet100 train splits from: {train_patterns}")
            datasets: List[Dataset] = [
                ImageFolder(root=folder, transform=transform)
                for folder in train_patterns
            ]
            concated = ConcatDataset(datasets)
            logger.info(f"Loaded ImageNet100 train split with {len(concated)} samples")
            return concated
        else:
            # Validation split
            val_path = os.path.join(data_path, 'val.X')
            if not os.path.exists(val_path):
                raise FileNotFoundError(
                    f"ImageNet100 val folder not found at: {val_path}\n"
                    f"Please download the dataset from Kaggle:\n"
                    f"  kaggle datasets download -d ambityga/imagenet100\n"
                    f"Then extract it to: {data_path}"
                )

            logger.info(f"Loading ImageNet100 val split from: {val_path}")
            val_dataset = ImageFolder(root=val_path, transform=transform)
            logger.info(f"Loaded ImageNet100 val split with {len(val_dataset)} samples")
            return val_dataset
    elif dataset_name == 'amrgraph':
        dataset = AMRGraphDataset(amr_file=data_path, bfs_depth=bfs_depth)
        logger.info(f"Loaded AMRGraphDataset with {len(dataset)} samples from {data_path}")
        return dataset
    elif dataset_name == 'amrgraph-global':
        dataset = AMRGraphDatasetGlobal(amr_file=data_path)
        logger.info(f"Loaded AMRGraphDatasetGlobal with {len(dataset)} samples from {data_path}")
        return dataset
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: imagenette, imagenet100, amrgraph, amrgraph-global"
        )


def create_train_val_test_splits(
    dataset: Dataset,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 0
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
    if seed:
        generator = torch.Generator().manual_seed(seed)
    else: 
        generator = torch.Generator()

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    if test_size == 0:
        logger.warning("Test split is empty. Consider adjusting train_split and val_split.")
        test_dataset = None

    return train_dataset, val_dataset, test_dataset