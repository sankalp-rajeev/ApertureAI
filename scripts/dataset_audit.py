"""
Dataset Audit Script for NEU Surface Defect Dataset

Task 1.1 — Validates:
- 6 defect classes present
- 300 images per class (1,800 total)
- Image dimensions: 200×200 grayscale
- No corrupted files
- Class balance report

Usage:
    python scripts/dataset_audit.py --data_dir data/raw/NEU-DET

Expected NEU-DET structure:
    NEU-DET/
    ├── train/
    │   └── images/
    │       ├── Cr_1.bmp
    │       ├── In_1.bmp
    │       └── ...
    └── ANNOTATIONS/
        └── *.xml (PASCAL VOC format)

Or alternative flat structure:
    NEU-DET/
    ├── Crazing/
    │   ├── Cr_1.bmp
    │   └── ...
    ├── Inclusion/
    └── ...
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Expected class names and prefixes
CLASS_PREFIXES = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted_Surface",
    "RS": "Rolled-in_Scale",
    "Sc": "Scratches",
}

EXPECTED_CLASSES = 6
EXPECTED_PER_CLASS = 300
EXPECTED_TOTAL = 1800
EXPECTED_SIZE = (200, 200)


def find_images(data_dir: Path) -> List[Path]:
    """Find all image files in the dataset directory."""
    extensions = {".bmp", ".jpg", ".jpeg", ".png"}
    images = []

    for ext in extensions:
        images.extend(data_dir.rglob(f"*{ext}"))
        images.extend(data_dir.rglob(f"*{ext.upper()}"))

    return sorted(set(images))


def get_class_from_filename(filename: str) -> str:
    """Extract class label from NEU filename convention (e.g., 'Cr_1.bmp' -> 'Crazing')."""
    # Try prefix matching first (Cr_, In_, Pa_, PS_, RS_, Sc_)
    for prefix, class_name in CLASS_PREFIXES.items():
        if filename.startswith(prefix + "_"):
            return class_name

    # Fallback: might be in class-named subdirectory
    return "Unknown"


def get_class_from_path(image_path: Path) -> str:
    """Get class from either filename or parent directory."""
    # First try filename
    class_name = get_class_from_filename(image_path.stem)
    if class_name != "Unknown":
        return class_name

    # Check if parent directory is a class name
    parent = image_path.parent.name
    for class_name in CLASS_PREFIXES.values():
        if parent.lower() == class_name.lower().replace("_", "").replace("-", ""):
            return class_name
        if parent.lower() == class_name.lower():
            return class_name

    # Check grandparent for nested structures
    if image_path.parent.parent.exists():
        grandparent = image_path.parent.parent.name
        for class_name in CLASS_PREFIXES.values():
            if grandparent.lower() == class_name.lower():
                return class_name

    return "Unknown"


def validate_image(image_path: Path) -> Tuple[bool, str, Tuple[int, int]]:
    """
    Validate a single image file.

    Returns:
        (is_valid, error_message, image_shape)
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            return False, "Failed to load (corrupted or unsupported format)", (0, 0)

        h, w = img.shape[:2]

        if (h, w) != EXPECTED_SIZE:
            return False, f"Wrong size: {w}x{h} (expected 200x200)", (h, w)

        return True, "", (h, w)

    except Exception as e:
        return False, f"Exception: {str(e)}", (0, 0)


def run_audit(data_dir: Path, verbose: bool = True) -> Dict:
    """
    Run full dataset audit.

    Returns:
        Audit results dictionary
    """
    print(f"\n{'='*60}")
    print("NEU Surface Defect Dataset Audit")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print()

    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return {"status": "FAILED", "error": "Directory not found"}

    # Find all images
    images = find_images(data_dir)
    print(f"Found {len(images)} image files")
    print()

    if len(images) == 0:
        print("ERROR: No images found!")
        return {"status": "FAILED", "error": "No images found"}

    # Classify and validate images
    class_counts = defaultdict(int)
    errors = []
    size_issues = []

    for img_path in images:
        class_name = get_class_from_path(img_path)
        class_counts[class_name] += 1

        is_valid, error_msg, shape = validate_image(img_path)

        if not is_valid:
            errors.append((img_path, error_msg))
            if verbose:
                print(f"  ERROR: {img_path.name} - {error_msg}")
        elif shape != EXPECTED_SIZE:
            size_issues.append((img_path, shape))

    # Print class distribution
    print("\nClass Distribution:")
    print("-" * 40)
    total = 0
    class_status = {}

    for class_name in sorted(CLASS_PREFIXES.values()):
        count = class_counts.get(class_name, 0)
        total += count
        status = "✓" if count == EXPECTED_PER_CLASS else "✗"
        class_status[class_name] = count
        print(f"  {class_name:20} : {count:4} images  {status}")

    unknown_count = class_counts.get("Unknown", 0)
    if unknown_count > 0:
        print(f"  {'Unknown':20} : {unknown_count:4} images  (unclassified)")
        total += unknown_count

    print("-" * 40)
    print(f"  {'TOTAL':20} : {total:4} images")
    print()

    # Summary
    print("Validation Summary:")
    print("-" * 40)

    checks = {
        "Total images": (total, EXPECTED_TOTAL, total == EXPECTED_TOTAL),
        "Number of classes": (len([c for c in class_counts if c != "Unknown"]), EXPECTED_CLASSES,
                              len([c for c in class_counts if c != "Unknown"]) == EXPECTED_CLASSES),
        "Corrupted files": (len(errors), 0, len(errors) == 0),
        "Size issues": (len(size_issues), 0, len(size_issues) == 0),
    }

    all_passed = True
    for check_name, (actual, expected, passed) in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        print(f"  {check_name:20} : {actual} (expected: {expected})  {status}")

    print()
    print("=" * 60)
    if all_passed:
        print("AUDIT PASSED - Dataset is ready for use")
    else:
        print("AUDIT FAILED - Please review errors above")
    print("=" * 60)

    return {
        "status": "PASSED" if all_passed else "FAILED",
        "total_images": total,
        "class_counts": dict(class_counts),
        "errors": errors,
        "size_issues": size_issues,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit NEU Surface Defect Dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/NEU-DET",
        help="Path to NEU-DET dataset directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error messages"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    results = run_audit(data_dir, verbose=args.verbose)

    # Exit with appropriate code
    sys.exit(0 if results["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
