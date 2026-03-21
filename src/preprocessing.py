"""
Module 1: OpenCV Preprocessing Pipeline

CLAHE normalization, adaptive thresholding, morphological operations,
and contour-guided ROI extraction for defect detection.
"""

import cv2
import numpy as np


def preprocess(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Preprocess a raw defect image to extract the ROI.

    Args:
        img: Input grayscale image (200x200)

    Returns:
        roi_tensor: Preprocessed ROI tensor
        contour_found: Whether a valid contour was detected
    """
    # TODO: Implement CLAHE, adaptive threshold, morphological ops, contour extraction
    raise NotImplementedError("Task 1.2")
