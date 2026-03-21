"""
Module 3: Optical Degradation Simulator.

Physics-based simulation of imaging system degradations:
- PSF convolution (Airy disk diffraction blur)
- Sensor noise (Poisson shot noise + Gaussian read noise)
- Illumination non-uniformity (vignetting)
- Depth-of-field blur

Used for DOE analysis to identify critical imaging parameters.

Reference: PRD Section 7.3 — Optical Degradation Simulator

Physics Foundation:
- PSF: Airy disk radius ∝ λ/(2·NA). For λ=550nm, NA 0.1–0.5 → σ=1–8px
- Shot noise: Poisson distributed, variance = signal level
- Read noise: Gaussian additive, signal-independent
- Vignetting: I(r) = I₀·cos⁴(α) for off-axis rays
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage


# Physics constants
WAVELENGTH_NM = 550  # Green light
PIXEL_SIZE_UM = 0.5  # PCB inspection camera pixel pitch


def compute_psf_sigma(na: float, wavelength_nm: float = WAVELENGTH_NM,
                      pixel_size_um: float = PIXEL_SIZE_UM) -> float:
    """
    Compute PSF sigma from numerical aperture.

    Formula: sigma_px = 0.61 * λ / (NA * pixel_size)

    For λ=550nm, pixel_size=0.5μm:
        NA=0.5 → σ ≈ 1.3 px
        NA=0.2 → σ ≈ 3.4 px
        NA=0.1 → σ ≈ 6.7 px

    Args:
        na: Numerical aperture (0.1 to 0.5 typical)
        wavelength_nm: Wavelength in nanometers
        pixel_size_um: Pixel pitch in micrometers

    Returns:
        sigma: PSF sigma in pixels
    """
    wavelength_um = wavelength_nm / 1000
    sigma_px = 0.61 * wavelength_um / (na * pixel_size_um)
    return sigma_px


def apply_psf(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply point spread function (Gaussian approximation of Airy disk).

    Physics: PSF radius proportional to lambda / (2 * NA)
    For lambda=550nm: NA=0.5 -> sigma~1px, NA=0.1 -> sigma~6px

    Args:
        img: Input image (uint8 or float)
        sigma: PSF sigma in pixels (0 = no blur)

    Returns:
        Blurred image (same dtype as input)
    """
    if sigma <= 0:
        return img.copy()

    # Store original dtype
    original_dtype = img.dtype

    # Convert to float for processing
    img_float = img.astype(np.float64)

    # Apply Gaussian blur (faster than scipy for 2D)
    # Kernel size should be 6*sigma to capture 99.7% of the distribution
    ksize = int(np.ceil(sigma * 6))
    if ksize % 2 == 0:
        ksize += 1  # Must be odd

    blurred = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)

    # Restore dtype
    if original_dtype == np.uint8:
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    return blurred


def apply_noise(img: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Apply sensor noise model (Poisson + Gaussian).

    Physics:
    - Poisson (shot noise): dominant at low light, variance = signal
    - Gaussian (read noise): additive, signal-independent

    SNR definition: SNR_dB = 20 * log10(signal / noise_std)

    Args:
        img: Input image (0-255 range, uint8 or float)
        snr_db: Signal-to-noise ratio in dB
                40dB = high quality, 30dB = medium, 20dB = low/noisy

    Returns:
        Noisy image (same dtype as input)
    """
    original_dtype = img.dtype

    # Convert to float [0, 1]
    if img.max() > 1:
        img_float = img.astype(np.float64) / 255.0
    else:
        img_float = img.astype(np.float64)

    # Compute noise standard deviation from SNR
    # SNR_dB = 20 * log10(signal / noise)
    # noise_std = signal / 10^(SNR_dB/20)
    signal_mean = np.mean(img_float) + 1e-8  # Avoid division by zero
    noise_std = signal_mean / (10 ** (snr_db / 20))

    # Poisson noise (shot noise) - proportional to signal level
    # Scale image to simulate photon counts, apply Poisson, scale back
    photon_scale = 1.0 / (noise_std ** 2 + 1e-8)
    photon_scale = np.clip(photon_scale, 1, 10000)

    # Poisson expects positive integers
    img_photons = np.clip(img_float * photon_scale, 0, None)
    noisy_photons = np.random.poisson(img_photons.astype(np.float64))
    img_poisson = noisy_photons / photon_scale

    # Gaussian noise (read noise) - additive
    gaussian_noise = np.random.normal(0, noise_std * 0.3, img_float.shape)
    img_noisy = img_poisson + gaussian_noise

    # Clip to valid range
    img_noisy = np.clip(img_noisy, 0, 1)

    # Restore original dtype
    if original_dtype == np.uint8:
        return (img_noisy * 255).astype(np.uint8)

    return img_noisy.astype(original_dtype)


def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Apply radial vignetting (illumination non-uniformity).

    Physics: I(r) = I0 * cos^4(alpha) for off-axis rays
    Simplified model: multiplicative radial falloff from center

    Args:
        img: Input image
        strength: Vignette strength (0=uniform, 0.10=mild, 0.25=strong)

    Returns:
        Vignetted image (same dtype as input)
    """
    if strength <= 0:
        return img.copy()

    original_dtype = img.dtype
    h, w = img.shape[:2]

    # Create radial distance map from center
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2

    # Normalized distance from center (0 at center, 1 at corners)
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    dist_normalized = dist / max_dist

    # Vignette mask: 1 at center, (1-strength) at corners
    # Use cos^4 falloff for physical accuracy
    vignette_mask = 1.0 - strength * (dist_normalized ** 2)
    vignette_mask = np.clip(vignette_mask, 1 - strength, 1)

    # Apply mask
    img_float = img.astype(np.float64)
    if len(img.shape) == 3:
        vignette_mask = vignette_mask[:, :, np.newaxis]

    vignetted = img_float * vignette_mask

    # Restore dtype
    if original_dtype == np.uint8:
        return np.clip(vignetted, 0, 255).astype(np.uint8)

    return vignetted.astype(original_dtype)


def apply_dof_blur(img: np.ndarray, z_offset: float) -> np.ndarray:
    """
    Apply depth-of-field blur (circle of confusion).

    Physics: Circle of confusion diameter increases linearly with
    distance from focal plane.

    Args:
        img: Input image
        z_offset: Distance from focal plane in arbitrary units
                  (0 = in focus, larger = more blur)

    Returns:
        Defocused image (same dtype as input)
    """
    if z_offset <= 0:
        return img.copy()

    # Circle of confusion modeled as disk blur
    # For simplicity, use Gaussian with sigma proportional to z_offset
    sigma = z_offset * 2.0  # Scale factor

    return apply_psf(img, sigma)


def apply_all_degradations(
    img: np.ndarray,
    psf_sigma: float = 0,
    snr_db: float = 40,
    vignette_strength: float = 0,
    dof_offset: float = 0
) -> np.ndarray:
    """
    Apply all degradations in sequence.

    Order: PSF -> DOF -> Noise -> Vignette
    (Blur before noise for physical accuracy)

    Args:
        img: Input image
        psf_sigma: PSF blur sigma in pixels
        snr_db: Signal-to-noise ratio in dB
        vignette_strength: Vignetting strength (0-0.25)
        dof_offset: Depth-of-field offset

    Returns:
        Degraded image
    """
    result = img.copy()

    # Optical blurs first (before sensor)
    if psf_sigma > 0:
        result = apply_psf(result, psf_sigma)

    if dof_offset > 0:
        result = apply_dof_blur(result, dof_offset)

    # Sensor noise
    if snr_db < 50:  # Only apply if meaningful noise level
        result = apply_noise(result, snr_db)

    # Illumination effects
    if vignette_strength > 0:
        result = apply_vignette(result, vignette_strength)

    return result


# DOE Configuration
DOE_PSF_LEVELS = [1, 3, 6]  # pixels (NA ≈ 0.5, 0.2, 0.1)
DOE_SNR_LEVELS = [40, 30, 20]  # dB (high, medium, low quality)
DOE_VIGNETTE_LEVELS = [0, 0.10, 0.25]  # strength (uniform, mild, strong)


def get_doe_configurations():
    """
    Generate all 27 DOE configurations (3 factors × 3 levels).

    Returns:
        List of dicts with keys: psf_sigma, snr_db, vignette_strength
    """
    import itertools

    configs = []
    for psf, snr, vig in itertools.product(
        DOE_PSF_LEVELS, DOE_SNR_LEVELS, DOE_VIGNETTE_LEVELS
    ):
        configs.append({
            "psf_sigma": psf,
            "snr_db": snr,
            "vignette_strength": vig
        })

    return configs


def validate_degradation_functions():
    """
    Unit tests for degradation functions.

    Returns:
        bool: True if all tests pass
    """
    print("Running degradation function unit tests...")

    # Create test image
    test_img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)

    # Test 1: apply_psf with sigma=0 returns identical image
    result = apply_psf(test_img, sigma=0)
    assert np.array_equal(result, test_img), "PSF sigma=0 should return identical image"
    print("  [PASS] apply_psf(sigma=0) returns identical image")

    # Test 2: apply_psf with sigma>0 changes image
    result = apply_psf(test_img, sigma=3)
    assert not np.array_equal(result, test_img), "PSF sigma>0 should blur image"
    print("  [PASS] apply_psf(sigma=3) changes image")

    # Test 3: apply_vignette with strength=0 returns identical image
    result = apply_vignette(test_img, strength=0)
    assert np.array_equal(result, test_img), "Vignette strength=0 should return identical image"
    print("  [PASS] apply_vignette(strength=0) returns identical image")

    # Test 4: apply_vignette darkens corners (use uniform image to isolate effect)
    uniform_img = np.full((100, 100), 128, dtype=np.uint8)
    result = apply_vignette(uniform_img, strength=0.25)
    center_val = result[50, 50]
    corner_val = result[0, 0]
    assert corner_val < center_val, "Vignette should darken corners"
    print("  [PASS] apply_vignette(strength=0.25) darkens corners")

    # Test 5: apply_noise changes image
    result = apply_noise(test_img, snr_db=20)
    assert not np.array_equal(result, test_img), "Noise should change image"
    print("  [PASS] apply_noise(snr_db=20) changes image")

    # Test 6: apply_dof_blur with offset=0 returns identical image
    result = apply_dof_blur(test_img, z_offset=0)
    assert np.array_equal(result, test_img), "DOF offset=0 should return identical image"
    print("  [PASS] apply_dof_blur(z_offset=0) returns identical image")

    # Test 7: Output dtype matches input dtype
    result = apply_all_degradations(test_img, psf_sigma=2, snr_db=30, vignette_strength=0.1)
    assert result.dtype == test_img.dtype, "Output dtype should match input dtype"
    print("  [PASS] Output dtype matches input dtype")

    # Test 8: DOE configurations
    configs = get_doe_configurations()
    assert len(configs) == 27, f"Expected 27 DOE configs, got {len(configs)}"
    print("  [PASS] get_doe_configurations() returns 27 configs")

    print("\nAll unit tests passed!")
    return True


if __name__ == "__main__":
    validate_degradation_functions()

    # Demo PSF physics
    print("\nPSF sigma values for different NA:")
    for na in [0.5, 0.3, 0.2, 0.1]:
        sigma = compute_psf_sigma(na)
        print(f"  NA={na:.1f} -> σ={sigma:.2f} px")
