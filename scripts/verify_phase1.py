"""
Phase 1 Verification Script

Run this after downloading the NEU dataset to verify:
1. Dataset audit passes
2. Preprocessing pipeline works
3. DataLoader produces correct tensor shapes
4. Augmentations are applied correctly

Usage:
    python scripts/verify_phase1.py --data_dir data/raw/NEU-DET
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch


def verify_preprocessing(data_dir: Path) -> bool:
    """Test preprocessing pipeline on sample images."""
    from src.preprocessing import preprocess, preprocess_for_tensor, visualize_preprocessing

    print("\n--- Testing Preprocessing Pipeline ---")

    # Find a sample image
    extensions = [".bmp", ".jpg", ".png"]
    sample_img = None
    for ext in extensions:
        files = list(data_dir.rglob(f"*{ext}"))
        if files:
            sample_img = files[0]
            break

    if sample_img is None:
        print("ERROR: No images found for preprocessing test")
        return False

    print(f"Testing on: {sample_img.name}")

    # Load image
    img = cv2.imread(str(sample_img), cv2.IMREAD_GRAYSCALE)
    print(f"  Input shape: {img.shape}")

    # Test basic preprocess
    processed, contour_found = preprocess(img)
    print(f"  Output shape: {processed.shape}")
    print(f"  Output dtype: {processed.dtype}")
    print(f"  Output range: [{processed.min():.3f}, {processed.max():.3f}]")
    print(f"  Contour found: {contour_found}")

    # Test tensor output
    tensor, _ = preprocess_for_tensor(img)
    print(f"  Tensor shape (CHW): {tensor.shape}")

    # Expected: (3, 224, 224), float32, [0, 1]
    if tensor.shape != (3, 224, 224):
        print(f"ERROR: Expected shape (3, 224, 224), got {tensor.shape}")
        return False

    if tensor.dtype != np.float32:
        print(f"ERROR: Expected float32, got {tensor.dtype}")
        return False

    # Save visualization
    output_dir = Path("results/phase1_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    vis = visualize_preprocessing(img, str(output_dir / "preprocessing_stages.png"))
    print(f"  Saved visualization to {output_dir / 'preprocessing_stages.png'}")

    print("✓ Preprocessing pipeline OK")
    return True


def verify_dataset(data_dir: Path) -> bool:
    """Test dataset loading and splits."""
    from src.dataset import NEUDataset, CLASS_NAMES

    print("\n--- Testing Dataset Class ---")

    try:
        train_dataset = NEUDataset(str(data_dir), split="train")
        val_dataset = NEUDataset(str(data_dir), split="val")
        test_dataset = NEUDataset(str(data_dir), split="test")
    except Exception as e:
        print(f"ERROR: Failed to create datasets: {e}")
        return False

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} images")

    # Verify split ratios (70/15/15)
    total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    train_ratio = len(train_dataset) / total
    val_ratio = len(val_dataset) / total
    test_ratio = len(test_dataset) / total

    print(f"  Ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")

    if not (0.68 <= train_ratio <= 0.72):
        print(f"WARNING: Train ratio {train_ratio:.2f} not close to 0.70")

    # Test loading a batch
    print("\n  Loading sample batch...")
    img, label = train_dataset[0]
    print(f"  Sample image shape: {img.shape}")
    print(f"  Sample label: {label} ({CLASS_NAMES[label]})")

    if img.shape != torch.Size([3, 224, 224]):
        print(f"ERROR: Expected shape [3, 224, 224], got {img.shape}")
        return False

    print("✓ Dataset class OK")
    return True


def verify_dataloader(data_dir: Path) -> bool:
    """Test DataLoader with batching."""
    from src.dataset import get_dataloaders
    from src.augmentations import get_train_transforms

    print("\n--- Testing DataLoader ---")

    train_loader, val_loader, test_loader = get_dataloaders(
        str(data_dir),
        batch_size=32,
        num_workers=0,  # Use 0 for testing
        train_transform=get_train_transforms()
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Load one batch
    print("\n  Loading one training batch...")
    batch_images, batch_labels = next(iter(train_loader))
    print(f"  Batch images shape: {batch_images.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")

    expected_shape = torch.Size([32, 3, 224, 224])
    if batch_images.shape != expected_shape:
        print(f"ERROR: Expected shape {expected_shape}, got {batch_images.shape}")
        return False

    print("✓ DataLoader OK")
    return True


def verify_augmentations(data_dir: Path) -> bool:
    """Test augmentation pipeline."""
    from src.augmentations import (
        get_train_transforms,
        get_val_transforms,
        visualize_augmentations
    )

    print("\n--- Testing Augmentations ---")

    # Load a sample image
    sample_img = next(data_dir.rglob("*.bmp"), None)
    if sample_img is None:
        sample_img = next(data_dir.rglob("*.jpg"), None)

    if sample_img is None:
        print("WARNING: No images found for augmentation test")
        return True

    img = cv2.imread(str(sample_img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=-1)  # HWC format

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    # Apply augmentations
    aug_train = train_transform(image=img)["image"]
    aug_val = val_transform(image=img)["image"]

    print(f"  Original shape: {img.shape}")
    print(f"  Train aug shape: {aug_train.shape}")
    print(f"  Val aug shape: {aug_val.shape}")

    # Save visualization
    output_dir = Path("results/phase1_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = visualize_augmentations(
        img,
        train_transform,
        n_samples=8,
        save_path=str(output_dir / "augmentation_samples.png")
    )
    print(f"  Saved augmentation visualization to {output_dir / 'augmentation_samples.png'}")

    print("✓ Augmentations OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 1 implementation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/NEU-DET",
        help="Path to NEU-DET dataset"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Phase 1 Verification")
    print("=" * 60)
    print(f"Data directory: {data_dir}")

    if not data_dir.exists():
        print(f"\nERROR: Dataset directory not found: {data_dir}")
        print("\nPlease download the NEU dataset first:")
        print("  1. Download from: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
        print("  2. Extract to: data/raw/NEU-DET/")
        print("  3. Run: python scripts/dataset_audit.py --data_dir data/raw/NEU-DET")
        sys.exit(1)

    # Run dataset audit first
    print("\n--- Running Dataset Audit ---")
    from scripts.dataset_audit import run_audit
    audit_results = run_audit(data_dir, verbose=False)

    if audit_results["status"] != "PASSED":
        print("\nERROR: Dataset audit failed. Fix issues before proceeding.")
        sys.exit(1)

    # Run verification tests
    all_passed = True

    all_passed &= verify_preprocessing(data_dir)
    all_passed &= verify_dataset(data_dir)
    all_passed &= verify_dataloader(data_dir)
    all_passed &= verify_augmentations(data_dir)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ PHASE 1 VERIFICATION PASSED")
        print("  Ready to proceed to Phase 2 (Classifier Training)")
    else:
        print("✗ PHASE 1 VERIFICATION FAILED")
        print("  Please review errors above")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
