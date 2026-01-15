"""Dataset wrappers and utilities."""

import os
import glob
import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
import torchvision
from torchvision.datasets import ImageFolder
from typing import Tuple, Optional, Callable, List
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

    Example:
        >>> from dino.data.transforms import DINOTransform
        >>> transform = DINOTransform()
        >>> dataset = get_dataset('imagenette', './data', transform=transform)
    """
    dataset_name = dataset_name.lower()

    elif dataset_name == 'imagenette':
        # Imagenette uses 'train' or 'val' as split argument
        split = 'train' if train else 'val'
        imagenette = torchvision.datasets.Imagenette(
            root=data_path,
            split=split,
            download=download,
            transform=transform
        )
        logger.info(f"Loaded Imagenette dataset ({split}) with {len(imagenette)} samples")
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
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: imagenette, imagenet100"
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


# =============================================================================
# Streaming Dataset Support (HuggingFace)
# =============================================================================

class HuggingFaceStreamingDataset(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset wrapper for HuggingFace streaming datasets.

    This allows using HuggingFace datasets with streaming=True in PyTorch
    DataLoaders without downloading the entire dataset.

    Args:
        hf_dataset: HuggingFace IterableDataset (from load_dataset with streaming=True)
        transform: Transform to apply to images (e.g., DINOTransform)
        image_key: Key for image column in the dataset (default: 'image')
        label_key: Key for label column in the dataset (default: 'label')

    Example:
        >>> from datasets import load_dataset
        >>> from dino.data.transforms import DINOTransform
        >>> hf_ds = load_dataset("ilee0022/ImageNet100", split="train", streaming=True)
        >>> transform = DINOTransform()
        >>> dataset = HuggingFaceStreamingDataset(hf_ds, transform)
        >>> for views, label in dataset:
        ...     print(len(views))  # 8
        ...     break
    """

    def __init__(
        self,
        hf_dataset,
        transform: Optional[Callable] = None,
        image_key: str = 'image',
        label_key: str = 'label'
    ):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __iter__(self):
        for sample in self.hf_dataset:
            image = sample[self.image_key]
            label = sample[self.label_key]

            # HuggingFace returns PIL images directly
            if self.transform is not None:
                image = self.transform(image)

            yield image, label


def get_streaming_dataset(
    dataset_name: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    seed: int = 42
):
    """
    Get a streaming dataset from HuggingFace Hub.

    This allows training on large datasets without downloading them entirely.
    Data is streamed on-the-fly during training.

    Supported streaming datasets:
    - imagenet100: ImageNet100 from HuggingFace (ilee0022/ImageNet100)

    Args:
        dataset_name: Name of the dataset
        split: Dataset split ('train', 'validation', 'test')
        transform: Transform to apply to images
        shuffle: Whether to shuffle the dataset (uses buffer shuffling)
        shuffle_buffer_size: Size of the shuffle buffer
        seed: Random seed for shuffling

    Returns:
        HuggingFaceStreamingDataset instance (PyTorch IterableDataset)

    Example:
        >>> from dino.data.transforms import DINOTransform
        >>> transform = DINOTransform()
        >>> dataset = get_streaming_dataset('imagenet100', 'train', transform)
        >>> dataloader = DataLoader(dataset, batch_size=32)
        >>> for views, labels in dataloader:
        ...     print(len(views))  # 8
        ...     break
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library is required for streaming. "
            "Install it with: pip install datasets"
        )

    dataset_name = dataset_name.lower()

    # Map dataset names to HuggingFace Hub identifiers
    hf_dataset_map = {
        'imagenet100': 'ilee0022/ImageNet100',
    }

    if dataset_name not in hf_dataset_map:
        raise ValueError(
            f"Unknown streaming dataset: {dataset_name}. "
            f"Supported streaming datasets: {list(hf_dataset_map.keys())}"
        )

    hf_dataset_id = hf_dataset_map[dataset_name]
    logger.info(f"Loading streaming dataset: {hf_dataset_id} (split={split})")

    # Load with streaming
    hf_dataset = load_dataset(
        hf_dataset_id,
        split=split,
        streaming=True,
        trust_remote_code=True
    )

    # Apply shuffle if requested (buffer-based shuffling for streaming)
    if shuffle:
        hf_dataset = hf_dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    logger.info(f"Streaming dataset ready: {hf_dataset_id}")

    return HuggingFaceStreamingDataset(
        hf_dataset=hf_dataset,
        transform=transform,
        image_key='image',
        label_key='label'
    )
