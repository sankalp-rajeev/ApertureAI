"""
PyTorch Dataset class for NEU Surface Defect Database.

Handles stratified train/val/test splits and augmentation pipeline.

Reference: PRD Section 5 — Dataset
- 1,800 images total (300 per class)
- 6 classes: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
- Stratified 70/15/15 split
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import preprocess_for_tensor


# Class mapping: prefix -> (class_name, class_index)
# Supports both naming conventions:
#   - Original: Cr_1.bmp, In_1.bmp, etc.
#   - Alternative: crazing_1.jpg, inclusion_1.jpg, etc.
CLASS_INFO = {
    # Original prefixes
    "Cr": ("Crazing", 0),
    "In": ("Inclusion", 1),
    "Pa": ("Patches", 2),
    "PS": ("Pitted_Surface", 3),
    "RS": ("Rolled-in_Scale", 4),
    "Sc": ("Scratches", 5),
    # Alternative naming (lowercase full names)
    "crazing": ("Crazing", 0),
    "inclusion": ("Inclusion", 1),
    "patches": ("Patches", 2),
    "pitted": ("Pitted_Surface", 3),
    "rolled": ("Rolled-in_Scale", 4),
    "scratches": ("Scratches", 5),
}

CLASS_NAMES = ["Crazing", "Inclusion", "Patches", "Pitted_Surface", "Rolled-in_Scale", "Scratches"]
NUM_CLASSES = 6


def get_class_from_filename(filename: str) -> Tuple[str, int]:
    """Extract class name and index from NEU filename."""
    filename_lower = filename.lower()

    # Try prefix matching (Cr_, In_, etc.)
    for prefix, (name, idx) in CLASS_INFO.items():
        if filename.startswith(prefix + "_") or filename_lower.startswith(prefix + "_"):
            return name, idx

    # Try full name matching (crazing_1, inclusion_1, etc.)
    for prefix, (name, idx) in CLASS_INFO.items():
        if filename_lower.startswith(prefix):
            return name, idx

    raise ValueError(f"Unknown class prefix in filename: {filename}")


def find_all_images(root: Path) -> List[Tuple[Path, int]]:
    """
    Find all images in the dataset directory.
    Handles both flat structure and class-subfolder structure.
    """
    extensions = {".bmp", ".jpg", ".jpeg", ".png"}
    images = []

    # Class folder name mapping
    CLASS_FOLDER_MAP = {
        "crazing": 0,
        "inclusion": 1,
        "patches": 2,
        "pitted_surface": 3,
        "rolled-in_scale": 4,
        "scratches": 5,
    }

    for ext in extensions:
        for img_path in root.rglob(f"*{ext}"):
            # Try to get class from parent folder name
            parent_name = img_path.parent.name.lower().replace("_", "_").replace("-", "-")
            if parent_name in CLASS_FOLDER_MAP:
                class_idx = CLASS_FOLDER_MAP[parent_name]
                images.append((img_path, class_idx))
                continue
            
            # Fallback: try filename prefix
            try:
                _, class_idx = get_class_from_filename(img_path.stem)
                images.append((img_path, class_idx))
            except ValueError:
                continue
    
    for ext in extensions:
        for img_path in root.rglob(f"*{ext.upper()}"):
            parent_name = img_path.parent.name.lower()
            if parent_name in CLASS_FOLDER_MAP:
                class_idx = CLASS_FOLDER_MAP[parent_name]
                if (img_path, class_idx) not in images:
                    images.append((img_path, class_idx))
                continue
            
            try:
                _, class_idx = get_class_from_filename(img_path.stem)
                if (img_path, class_idx) not in images:
                    images.append((img_path, class_idx))
            except ValueError:
                continue

    return sorted(set(images), key=lambda x: x[0])

def create_stratified_splits(
    images: List[Tuple[Path, int]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Tuple[Path, int]]]:
    """
    Create stratified train/val/test splits.

    Ensures each split has the same class distribution as the full dataset.

    Args:
        images: List of (path, class_idx) tuples
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        Dict with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)

    # Group by class
    class_groups: Dict[int, List[Tuple[Path, int]]] = {i: [] for i in range(NUM_CLASSES)}
    for img_path, class_idx in images:
        class_groups[class_idx].append((img_path, class_idx))

    splits = {"train": [], "val": [], "test": []}

    # Stratified split per class
    for class_idx, class_images in class_groups.items():
        random.shuffle(class_images)
        n = len(class_images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits["train"].extend(class_images[:n_train])
        splits["val"].extend(class_images[n_train:n_train + n_val])
        splits["test"].extend(class_images[n_train + n_val:])

    # Shuffle each split
    for split in splits.values():
        random.shuffle(split)

    return splits


class NEUDataset(Dataset):
    """
    NEU Surface Defect Dataset.

    Args:
        root: Path to dataset root directory
        split: One of 'train', 'val', 'test'
        transform: albumentations transform pipeline (applied after preprocessing)
        preprocess: Whether to apply OpenCV preprocessing pipeline
        cache_preprocessed: Whether to cache preprocessed images in memory
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        preprocess: bool = True,
        cache_preprocessed: bool = False,
        seed: int = 42
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.preprocess = preprocess
        self.cache_preprocessed = cache_preprocessed

        # Find all images and create splits
        all_images = find_all_images(self.root)
        if len(all_images) == 0:
            raise RuntimeError(f"No images found in {self.root}")

        splits = create_stratified_splits(all_images, seed=seed)
        self.images = splits[split]

        # Optional: cache preprocessed images
        self._cache: Dict[int, np.ndarray] = {}

        print(f"NEUDataset: {split} split with {len(self.images)} images")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution for this split."""
        counts = {i: 0 for i in range(NUM_CLASSES)}
        for _, class_idx in self.images:
            counts[class_idx] += 1

        print("  Class distribution:")
        for idx, count in sorted(counts.items()):
            print(f"    {CLASS_NAMES[idx]:20}: {count}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            image: (3, 224, 224) float32 tensor, normalized [0, 1]
            label: class index (0-5)
        """
        img_path, label = self.images[idx]

        # Check cache
        if self.cache_preprocessed and idx in self._cache:
            image = self._cache[idx]
        else:
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to load image: {img_path}")

            if self.preprocess:
                # Apply full preprocessing pipeline
                image, _ = preprocess_for_tensor(img)
            else:
                # Basic resize and normalize only
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                image = np.stack([img, img, img], axis=0)

            # Cache if enabled
            if self.cache_preprocessed:
                self._cache[idx] = image

        # Apply augmentations (expects HWC format)
        if self.transform is not None:
            # CHW -> HWC for albumentations
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc)
            image_hwc = augmented["image"]
            # HWC -> CHW
            image = np.transpose(image_hwc, (2, 0, 1))

        # Convert to tensor
        image_tensor = torch.from_numpy(image.copy()).float()

        return image_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced dataset handling.

        Returns:
            Tensor of shape (NUM_CLASSES,) with inverse frequency weights
        """
        counts = torch.zeros(NUM_CLASSES)
        for _, class_idx in self.images:
            counts[class_idx] += 1

        # Inverse frequency weighting
        weights = 1.0 / counts
        weights = weights / weights.sum() * NUM_CLASSES

        return weights


def get_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    preprocess: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root: Path to dataset root
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_transform: Augmentation for training
        val_transform: Augmentation for validation/test (usually None)
        preprocess: Whether to apply OpenCV preprocessing
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = NEUDataset(
        root, split="train", transform=train_transform,
        preprocess=preprocess, seed=seed
    )
    val_dataset = NEUDataset(
        root, split="val", transform=val_transform,
        preprocess=preprocess, seed=seed
    )
    test_dataset = NEUDataset(
        root, split="test", transform=val_transform,
        preprocess=preprocess, seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
