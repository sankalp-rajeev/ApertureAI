"""
Module 3: Optical Degradation Simulator.

Physics-based simulation of imaging system degradations:
- PSF convolution (Airy disk diffraction blur)
- Sensor noise (Poisson shot noise + Gaussian read noise)
- Illumination non-uniformity (vignetting)
- Depth-of-field blur

Used for DOE analysis to identify critical imaging parameters.
"""

import numpy as np
import cv2
from scipy import ndimage


def apply_psf(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply point spread function (Gaussian approximation of Airy disk).

    Physics: PSF radius proportional to lambda / (2 * NA)
    For lambda=550nm: NA=0.5 -> sigma~1px, NA=0.1 -> sigma~6px

    Args:
        img: Input image
        sigma: PSF sigma in pixels (0 = no blur)

    Returns:
        Blurred image
    """
    # TODO: Implement PSF convolution
    raise NotImplementedError("Task 3.1")


def apply_noise(img: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Apply sensor noise model (Poisson + Gaussian).

    Physics:
    - Poisson (shot noise): dominant at low light, variance = signal
    - Gaussian (read noise): additive, signal-independent

    Args:
        img: Input image (0-255 range)
        snr_db: Signal-to-noise ratio in dB (20=low, 40=high)

    Returns:
        Noisy image
    """
    # TODO: Implement noise model
    raise NotImplementedError("Task 3.1")


def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Apply radial vignetting (illumination non-uniformity).

    Physics: I(r) = I0 * cos^4(alpha) for off-axis rays

    Args:
        img: Input image
        strength: Vignette strength (0=uniform, 0.25=strong)

    Returns:
        Vignetted image
    """
    # TODO: Implement vignetting
    raise NotImplementedError("Task 3.1")


def apply_dof_blur(img: np.ndarray, z_offset: float) -> np.ndarray:
    """
    Apply depth-of-field blur (circle of confusion).

    Args:
        img: Input image
        z_offset: Distance from focal plane (arbitrary units)

    Returns:
        Defocused image
    """
    # TODO: Implement DOF blur
    raise NotImplementedError("Task 3.1")
