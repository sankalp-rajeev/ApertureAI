"""
Module 4b: SSIM Pre-filter for defect detection.

Compares incoming frames against a defect-free reference template
using Structural Similarity Index to skip clean frames and reduce
classifier load.

Reference: PRD Section 7.4 — SSIM Pre-filter
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM between two grayscale images using OpenCV.

    This avoids the scikit-image dependency by implementing SSIM directly
    with the standard formula using Gaussian-weighted statistics.

    Args:
        img1: First image (H, W) uint8
        img2: Second image (H, W) uint8

    Returns:
        SSIM score in [0, 1]
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def build_reference_template(clean_images: List[np.ndarray]) -> np.ndarray:
    """
    Build defect-free reference template from clean images.

    Uses the mean of confirmed defect-free images to create a stable
    reference that represents the expected "clean" surface appearance.

    Args:
        clean_images: List of grayscale clean image crops (H, W) uint8

    Returns:
        reference: Mean template image (H, W) uint8
    """
    if len(clean_images) == 0:
        raise ValueError("Need at least one clean image for reference template")

    # Use up to 10 images for stable reference
    images_to_use = clean_images[:10]

    # Resize all to common size
    target_h, target_w = images_to_use[0].shape[:2]
    resized = []
    for img in images_to_use:
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized.append(img.astype(np.float32))

    stack = np.stack(resized)
    return stack.mean(axis=0).astype(np.uint8)


def ssim_prefilter(img: np.ndarray, reference: np.ndarray,
                   threshold: float) -> bool:
    """
    Determine if a frame should be skipped (no defect detected).

    High SSIM = similar to clean reference = skip classifier.

    Args:
        img: Input grayscale image (H, W)
        reference: Clean reference template (H, W)
        threshold: SSIM threshold (frames above this are skipped)

    Returns:
        True if frame should be SKIPPED (looks clean)
    """
    # Ensure same size
    if img.shape[:2] != reference.shape[:2]:
        img = cv2.resize(img, (reference.shape[1], reference.shape[0]))

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    score = compute_ssim(img, reference)
    return score > threshold


def tune_threshold(
    test_images: List[np.ndarray],
    test_labels: List[int],
    reference: np.ndarray,
    clean_label: int = 0,
    thresholds: Optional[np.ndarray] = None,
) -> Dict:
    """
    Find the highest SSIM threshold where 0 defective images are skipped.

    Args:
        test_images: List of test images
        test_labels: List of corresponding labels
        reference: Clean reference template
        clean_label: Label for defect-free class (default: 0)
        thresholds: Array of thresholds to try

    Returns:
        Dict with best_threshold, skip_rate, missed_defects, all_results
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.95, 0.05)

    results = []
    for thresh in thresholds:
        skipped = 0
        missed = 0
        for img, label in zip(test_images, test_labels):
            if ssim_prefilter(img, reference, thresh):
                skipped += 1
                if label != clean_label:
                    missed += 1

        results.append({
            "threshold": float(thresh),
            "skip_rate": skipped / len(test_images) if test_images else 0,
            "skipped": skipped,
            "missed_defects": missed,
            "total": len(test_images),
        })

    # Find highest threshold with 0 misses
    safe = [r for r in results if r["missed_defects"] == 0]

    if safe:
        best = max(safe, key=lambda r: r["skip_rate"])
    else:
        # No safe threshold found — use lowest threshold
        best = min(results, key=lambda r: r["missed_defects"])

    return {
        "best_threshold": best["threshold"],
        "skip_rate": best["skip_rate"],
        "missed_defects": best["missed_defects"],
        "all_results": results,
    }


if __name__ == "__main__":
    print("SSIM Pre-filter module loaded successfully.")
    # Quick self-test
    img1 = np.random.randint(100, 200, (100, 100), dtype=np.uint8)
    img2 = img1.copy()
    score = compute_ssim(img1, img2)
    assert score > 0.99, f"Identical images should have SSIM ~1.0, got {score}"
    print(f"  [PASS] SSIM of identical images: {score:.4f}")

    img3 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    score2 = compute_ssim(img1, img3)
    assert score2 < score, "Different images should have lower SSIM"
    print(f"  [PASS] SSIM of different images: {score2:.4f} < {score:.4f}")

    ref = build_reference_template([img1, img1, img1])
    assert ref.shape == (100, 100), f"Reference shape wrong: {ref.shape}"
    print("  [PASS] Reference template built correctly")

    print("\nAll SSIM tests passed!")
