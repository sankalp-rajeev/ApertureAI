"""
Module 1: OpenCV Preprocessing Pipeline

CLAHE normalization, adaptive thresholding, morphological operations,
and contour-guided ROI extraction for defect detection.

Reference: PRD Section 7.1 — OpenCV Preprocessing Pipeline

Engineering Decisions:
- Adaptive thresholding (Gaussian) over global Otsu: handles non-uniform illumination
- CLAHE over global HE: limits noise amplification in flat background regions
- Morphological closing before opening: fills gaps then removes noise
- Contour-guided ROI over fixed center crop: handles off-center defects
"""

from typing import Tuple, Optional
import cv2
import numpy as np


# Default parameters from PRD Section 7.1
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2
CLOSING_KERNEL_SIZE = 5
OPENING_KERNEL_SIZE = 3
ROI_PADDING = 10
DEFAULT_SIZE = 200


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: Tuple[int, int] = CLAHE_TILE_GRID
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    CLAHE limits noise amplification in flat background regions compared to
    global histogram equalization. This is critical for industrial surfaces
    where background texture should not be over-enhanced.

    Args:
        img: Input grayscale image (uint8)
        clip_limit: Threshold for contrast limiting
        tile_grid: Size of grid for histogram equalization

    Returns:
        CLAHE-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def apply_adaptive_threshold(
    img: np.ndarray,
    block_size: int = ADAPTIVE_BLOCK_SIZE,
    C: int = ADAPTIVE_C
) -> np.ndarray:
    """
    Apply adaptive Gaussian thresholding.

    Adaptive thresholding handles non-uniform illumination better than
    global Otsu thresholding. The Gaussian method computes the threshold
    as a weighted sum of neighborhood values.

    Args:
        img: Input grayscale image (uint8)
        block_size: Size of pixel neighborhood for threshold calculation (must be odd)
        C: Constant subtracted from the weighted mean

    Returns:
        Binary thresholded image
    """
    return cv2.adaptiveThreshold(
        img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # Defects are typically darker
        blockSize=block_size,
        C=C
    )


def apply_morphological_ops(
    binary_img: np.ndarray,
    closing_ksize: int = CLOSING_KERNEL_SIZE,
    opening_ksize: int = OPENING_KERNEL_SIZE
) -> np.ndarray:
    """
    Apply morphological closing followed by opening.

    Closing (dilation then erosion): fills small gaps in defect boundaries
    Opening (erosion then dilation): removes small noise/artifacts

    Order matters: closing first preserves thin defect features (scratches)
    while filling gaps; opening then cleans up noise.

    Args:
        binary_img: Binary thresholded image
        closing_ksize: Kernel size for closing operation
        opening_ksize: Kernel size for opening operation

    Returns:
        Morphologically processed binary image
    """
    # Closing: fills gaps in defect boundaries
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (closing_ksize, closing_ksize)
    )
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, closing_kernel)

    # Opening: removes small noise
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (opening_ksize, opening_ksize)
    )
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, opening_kernel)

    return opened


def extract_roi(
    img: np.ndarray,
    binary_mask: np.ndarray,
    padding: int = ROI_PADDING,
    min_contour_area: int = 50
) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
    """
    Extract ROI using contour-guided bounding box.

    Finds the largest contour in the binary mask and extracts the
    corresponding region from the original image with padding.
    Falls back to center crop if no valid contour is found.

    Args:
        img: Original grayscale image
        binary_mask: Morphologically processed binary mask
        padding: Pixels to add around bounding box
        min_contour_area: Minimum contour area to consider valid

    Returns:
        roi: Extracted region of interest
        contour_found: Whether a valid contour was detected
        bbox: Bounding box (x, y, w, h) or None if fallback used
    """
    h, w = img.shape[:2]

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        # Fallback: center crop
        roi = center_crop(img)
        return roi, False, None

    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < min_contour_area:
        # Contour too small, use center crop
        roi = center_crop(img)
        return roi, False, None

    # Get bounding box with padding
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    # Apply padding with bounds checking
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    roi = img[y1:y2, x1:x2]

    return roi, True, (x1, y1, x2 - x1, y2 - y1)


def center_crop(
    img: np.ndarray,
    crop_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Extract center crop from image.

    Fallback when no valid contour is detected.

    Args:
        img: Input image
        crop_size: (height, width) of crop. If None, uses 80% of image size

    Returns:
        Center-cropped image
    """
    h, w = img.shape[:2]

    if crop_size is None:
        crop_h = int(h * 0.8)
        crop_w = int(w * 0.8)
    else:
        crop_h, crop_w = crop_size

    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2

    return img[start_y:start_y + crop_h, start_x:start_x + crop_w]


def resize_for_model(
    img: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Resize image to model input size.

    Uses INTER_LINEAR for upscaling, INTER_AREA for downscaling
    (better for reducing aliasing artifacts).

    Args:
        img: Input image
        target_size: (width, height) target dimensions

    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    if h * w > target_h * target_w:
        # Downscaling
        interpolation = cv2.INTER_AREA
    else:
        # Upscaling
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(img, target_size, interpolation=interpolation)


def normalize_for_model(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range for model input.

    Args:
        img: Input uint8 image

    Returns:
        Normalized float32 image
    """
    return img.astype(np.float32) / 255.0


def grayscale_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert grayscale to 3-channel for model input.

    EfficientNet expects 3 channels. We replicate the grayscale
    channel since the pretrained weights will be averaged anyway.

    Args:
        img: Grayscale image (H, W)

    Returns:
        3-channel image (H, W, 3)
    """
    return np.stack([img, img, img], axis=-1)


def preprocess(
    img: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    return_intermediate: bool = False
) -> Tuple[np.ndarray, bool]:
    """
    Full preprocessing pipeline for defect detection.

    Pipeline:
    1. CLAHE contrast enhancement
    2. Adaptive thresholding
    3. Morphological operations
    4. Contour-guided ROI extraction
    5. Resize to model input size
    6. Normalize and convert to 3-channel

    Args:
        img: Input grayscale image (200x200, uint8)
        target_size: Output size for model (width, height)
        return_intermediate: If True, return dict of intermediate results

    Returns:
        If return_intermediate=False:
            processed: Preprocessed image ready for model (H, W, 3), float32 [0,1]
            contour_found: Whether a valid contour was detected

        If return_intermediate=True:
            results: Dict containing all intermediate processing steps
    """
    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: CLAHE
    clahe_img = apply_clahe(img)

    # Step 2: Adaptive thresholding
    binary = apply_adaptive_threshold(clahe_img)

    # Step 3: Morphological operations
    morph = apply_morphological_ops(binary)

    # Step 4: ROI extraction
    roi, contour_found, bbox = extract_roi(img, morph)

    # Step 5: Resize
    resized = resize_for_model(roi, target_size)

    # Step 6: Normalize and convert to RGB
    normalized = normalize_for_model(resized)
    rgb = grayscale_to_rgb(normalized)

    if return_intermediate:
        return {
            "original": img,
            "clahe": clahe_img,
            "binary": binary,
            "morphological": morph,
            "roi": roi,
            "bbox": bbox,
            "resized": resized,
            "normalized": normalized,
            "final": rgb,
            "contour_found": contour_found,
        }

    return rgb, contour_found


def preprocess_for_tensor(
    img: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, bool]:
    """
    Preprocess and return tensor in CHW format for PyTorch.

    Args:
        img: Input grayscale image
        target_size: Output size (width, height)

    Returns:
        tensor: (3, H, W) float32 tensor
        contour_found: Whether contour was detected
    """
    processed, contour_found = preprocess(img, target_size)
    # HWC -> CHW
    tensor = np.transpose(processed, (2, 0, 1))
    return tensor, contour_found


def visualize_preprocessing(
    img: np.ndarray,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create visualization of preprocessing stages.

    Useful for quality assurance and debugging.

    Args:
        img: Input grayscale image
        save_path: If provided, save visualization to this path

    Returns:
        Visualization image showing all stages
    """
    results = preprocess(img, return_intermediate=True)

    # Create visualization grid
    stages = [
        ("Original", results["original"]),
        ("CLAHE", results["clahe"]),
        ("Binary", results["binary"]),
        ("Morphological", results["morphological"]),
        ("ROI", results["roi"]),
        ("Resized", (results["resized"] * 255).astype(np.uint8)),
    ]

    # Resize all to same size for display
    display_size = (200, 200)
    display_imgs = []

    for name, stage_img in stages:
        if len(stage_img.shape) == 2:
            display = cv2.resize(stage_img, display_size)
        else:
            display = cv2.resize(stage_img, display_size)
        display_imgs.append(display)

    # Stack horizontally
    vis = np.hstack(display_imgs)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis
