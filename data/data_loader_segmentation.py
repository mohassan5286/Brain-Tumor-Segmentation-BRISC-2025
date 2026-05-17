# ============================================================================
# Data Loading and Preprocessing Module - BRISC 2025
# ============================================================================
# This module handles loading brain MRI images and segmentation masks,
# applies data augmentation, and prepares batches for training and testing.
# ============================================================================

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

# ============================================================================
# Custom Dataset Class
# ============================================================================


class SegmentationDataset(Dataset):
    """Dataset class for loading paired MRI images and segmentation masks."""

    def __init__(self, image_dir, mask_dir, transform=None):
        # Store directories and transformations
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # List all image files in directory
        self.images = os.listdir(image_dir)

    def __len__(self):
        """Return total number of images in dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Load and return image, mask, and tumor type at given index."""
        # Construct file paths
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(
            self.mask_dir, self.images[idx].replace(".jpg", ".png")
        )

        # Load images in grayscale (single channel)
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Apply transformations if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Extract tumor type from filename (format: ..._TYPE_...)
        tumor_type = self.images[idx].split("_")[3]

        return image, mask, tumor_type


# ============================================================================
# Joint Transformation Class
# ============================================================================


class JointTransform:
    """Apply identical augmentations to both image and mask for consistency."""

    def __init__(self, base_transform, mask_transform=None):
        # Store transformations for image and mask
        self.base_transform = base_transform
        self.mask_transform = mask_transform if mask_transform else base_transform

    def __call__(self, image, mask):
        # Use same seed for reproducible augmentations
        seed = random.randint(0, 2**32)
        # Apply to image
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.base_transform(image)

        # Apply to mask with same seed
        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)

        return image, mask


# ============================================================================
# Training Data Loader Creation Function
# ============================================================================


def get_data_loader(data_dir, batch_size):
    """Create and return train and test data loaders with augmentation."""
    # Define transformations for training images
    base_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # Normalize using dataset statistics
            transforms.Normalize(mean=[0.16666122], std=[0.17487395]),
        ]
    )

    # Define transformations for training masks (interpolation differs)
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # Threshold masks to binary values
            transforms.Lambda(lambda x: (x > 0.5).float()),
        ]
    )

    # Combine image and mask transformations for training
    train_transform = JointTransform(base_transform, mask_transform)

    # Define transformations for test images (no augmentation)
    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.16666122], std=[0.17487395]),
        ]
    )

    # Define transformations for test masks
    test_mask_transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),
        ]
    )

    # Create training dataset
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "train", "images"),
        mask_dir=os.path.join(data_dir, "train", "masks"),
        transform=train_transform,
    )

    # Create test dataset
    test_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "test", "images"),
        mask_dir=os.path.join(data_dir, "test", "masks"),
        transform=JointTransform(test_transform, test_mask_transform),
    )

    # Create data loaders with shuffling for training
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader


# ============================================================================
# Testing Data Loader Creation Function
# ============================================================================


def get_test_loader(data_dir, batch_size):
    """Create and return test data loader without augmentation."""
    
    # Define transformations for test images (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Normalize using dataset statistics
        transforms.Normalize(mean=[0.16666122], std=[0.17487395]),
    ])

    # Define transformations for test masks
    test_mask_transform = transforms.Compose([
        transforms.Resize(
            (256, 256), interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.ToTensor(),
        # Threshold masks to binary values
        transforms.Lambda(lambda x: (x > 0.5).float()),
    ])

    # Create test dataset
    test_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "test", "images"),
        mask_dir=os.path.join(data_dir, "test", "masks"),
        transform=JointTransform(test_transform, test_mask_transform),
    )

    # Create data loader (no shuffling for evaluation)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return test_loader