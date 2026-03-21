"""
Augmentation pipelines for NEU Surface Defect Dataset.

Reference: PRD Section 5.3 — Augmentation Strategy

Augmentations mimic real inspection variability:
✅ Random rotation ±15° — part misalignment on conveyor
✅ Horizontal/vertical flip — symmetric defect morphology
✅ Random brightness ±20% — illumination variation between batches
✅ Gaussian blur (σ=0.5–1.5) — slight focus variation
✅ Elastic deformation — surface texture variation
❌ Random cropping that removes defect region
❌ Color jitter — these are grayscale images
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms() -> A.Compose:
    """
    Training augmentation pipeline.

    These augmentations mimic real-world inspection variability
    without introducing unrealistic transforms.
    """
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Illumination variation (brightness only, no color)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.1,
            p=0.5
        ),

        # Slight focus variation
        A.GaussianBlur(
            blur_limit=(3, 5),  # kernel size, corresponds to σ ~0.5-1.5
            p=0.3
        ),

        # Surface texture variation
        A.ElasticTransform(
            alpha=50,  # intensity
            sigma=10,  # smoothness
            p=0.2
        ),

        # Slight perspective changes (conveyor angle variation)
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            shear=(-5, 5),
            p=0.3
        ),

        # Minor noise (sensor variation)
        A.GaussNoise(
            std_range=(0.02, 0.08),  # std deviation range for noise
            p=0.2
        ),
    ])


def get_val_transforms() -> A.Compose:
    """
    Validation/test transform pipeline.

    No augmentation — only normalization for consistent evaluation.
    """
    return A.Compose([
        # No augmentations for validation
    ])


def get_heavy_train_transforms() -> A.Compose:
    """
    Heavier augmentation for experiments.

    Use this for Task 2.3 (hyperparameter tuning) to test
    augmentation intensity effects.
    """
    return A.Compose([
        # Stronger geometric transforms
        A.Rotate(limit=30, border_mode=0, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=0,
            p=0.5
        ),

        # Stronger illumination variation
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.2,
            p=0.7
        ),

        # More aggressive blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.4),

        # Stronger elastic deformation
        A.ElasticTransform(
            alpha=100,
            sigma=15,
            p=0.3
        ),

        # Grid distortion (surface warping)
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            p=0.2
        ),

        # More noise
        A.GaussNoise(
            std_range=(0.04, 0.15),  # std deviation range
            p=0.4
        ),
    ])


def get_light_train_transforms() -> A.Compose:
    """
    Light augmentation for experiments.

    Use this for Task 2.3 (hyperparameter tuning) to test
    minimal augmentation effects.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.05,
            p=0.3
        ),
    ])


def visualize_augmentations(
    image: np.ndarray,
    transform: A.Compose,
    n_samples: int = 8,
    save_path: str = None
) -> np.ndarray:
    """
    Visualize augmentation effects on a single image.

    Args:
        image: Input image (H, W, C) float32 [0, 1]
        transform: Augmentation pipeline
        n_samples: Number of augmented samples to show
        save_path: Optional path to save visualization

    Returns:
        Grid of augmented images
    """
    import cv2

    samples = [image]  # Original first

    for _ in range(n_samples - 1):
        augmented = transform(image=image)["image"]
        samples.append(augmented)

    # Convert to uint8 for display
    samples_uint8 = [(s * 255).astype(np.uint8) for s in samples]

    # Arrange in grid
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    # Resize all to same size
    h, w = samples_uint8[0].shape[:2]
    grid_rows = []

    for row_idx in range(n_rows):
        row_images = []
        for col_idx in range(n_cols):
            idx = row_idx * n_cols + col_idx
            if idx < len(samples_uint8):
                row_images.append(samples_uint8[idx])
            else:
                # Pad with black
                row_images.append(np.zeros_like(samples_uint8[0]))
        grid_rows.append(np.hstack(row_images))

    grid = np.vstack(grid_rows)

    if save_path:
        cv2.imwrite(save_path, grid)

    return grid
