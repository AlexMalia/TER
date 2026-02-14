"""DataLoader creation utilities."""

from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, Union
import logging
import torch


from .datasets import get_dataset, create_train_val_test_splits
from .transforms import DINOTransform

from ..config.config import DataConfig, AugmentationConfig

logger = logging.getLogger(__name__)


def collate_multi_crop(samples):
    """
    Custom collate function for multi-crop batches.

    Converts list of samples (views_list, label) tuples into proper batch format.

    Args:
        samples: List of (views_list, label) tuples

    Returns:
        Tuple of (views_batch, labels_batch) where:
            - views_batch is a list of tensors, one per view type
            - labels_batch is a tensor of labels

    Inputs : 
        [(List[Tensor[3, H, W]], label1),    # List of 8 tensors per sample
        (List[Tensor[3, H, W]], label2),
        ...]
        
        where:
            - List length = 8 (2 global + 6 local views)
            - H, W vary by view type:
                * Global views: H=224, W=224
                * Local views: H=96, W=96

    Outputs:
        (
            [Tensor[batch_size, 3, 224, 224],  # global view 1
            Tensor[batch_size, 3, 224, 224],  # global view 2
            Tensor[batch_size, 3, 96, 96],    # local view 1
            Tensor[batch_size, 3, 96, 96],    # local view 2
            Tensor[batch_size, 3, 96, 96],    # local view 3
            Tensor[batch_size, 3, 96, 96],    # local view 4
            Tensor[batch_size, 3, 96, 96],    # local view 5
            Tensor[batch_size, 3, 96, 96]],   # local view 6
            
            Tensor[batch_size]  # labels: [label1, label2, ...]
        )
    """

    # batch is a list of (views_list, label)
    # views_list contains [global1, global2, local1, ..., local6]

    views_lists = [item[0] for item in samples]
    labels = torch.tensor([item[1] for item in samples])

    # Transpose: from list of lists to list of batches
    # [[g1_img1, g2_img1, l1_img1, ...], [g1_img2, g2_img2, l2_img2, ...]]
    # -> [[g1_img1, g1_img2, ...], [g2_img1, g2_img2, ...], ...]
    num_views = len(views_lists[0])
    views_batch = []

    for view_idx in range(num_views):
        view_batch = torch.stack([views[view_idx] for views in views_lists])
        views_batch.append(view_batch)

    return views_batch, labels


def create_dataloaders(
    data_config: DataConfig,
    augmentation_config: AugmentationConfig,
    return_test: bool = False
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train, validation, and optionally test dataloaders from config.

    Supports both standard datasets (downloaded) and.

    Args:
        data_config: DataConfig instance
        augmentation_config: AugmentationConfig instance
        return_test: Whether to return test dataloader

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        If return_test is False, test_loader will be None
    """

    # Create transform
    transform = DINOTransform.from_config(augmentation_config)

    # Load dataset with transform
    dataset = get_dataset(
        dataset_name=data_config.dataset,
        data_path=data_config.data_path,
        transform=transform,
        download=True,
        train=True
    )

    logger.info(f"Loaded {data_config.dataset} dataset with {len(dataset)} samples")

    # Split into train/val/test
    train_dataset, val_dataset, test_dataset = create_train_val_test_splits(
        dataset,
        train_split=data_config.train_split,
        val_split=data_config.val_split,
        seed=data_config.seed
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.do_pin_memory,
        collate_fn=collate_multi_crop,
        drop_last=True  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.do_pin_memory,
        collate_fn=collate_multi_crop,
        drop_last=False
    ) if len(val_dataset) > 0 else None

    test_loader = None
    if return_test and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.do_pin_memory,
            collate_fn=collate_multi_crop,
            drop_last=False
        )

    logger.info(
        f"Created dataloaders: "
        f"train={len(train_loader)} batches, "
        f"val={len(val_loader) if val_loader else 0} batches, "
        f"test={len(test_loader) if test_loader else 0} batches"
    )

    return train_loader, val_loader, test_loader