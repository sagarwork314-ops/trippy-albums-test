"""Sharpness via Laplacian variance — the classic, well-validated focus metric.

A blurry photo has a smooth intensity surface, so its Laplacian (second
derivative) response is small and uniform -> low variance. A sharp photo has
strong edges everywhere -> high-variance Laplacian response. We normalize
the raw variance on a log scale because it spans orders of magnitude between
a flat sky and a richly textured scene.
"""
from __future__ import annotations

import numpy as np

# Variance of the Laplacian response above which we consider an image "tack
# sharp" for normalization purposes. Calibrated empirically against typical
# phone/camera photos; tune if your source material skews differently.
_REFERENCE_VARIANCE = 1500.0

_LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _convolve3x3(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid-mode 3x3 convolution without external dependencies."""
    out = np.zeros((gray.shape[0] - 2, gray.shape[1] - 2), dtype=np.float64)
    for ky in range(3):
        for kx in range(3):
            weight = kernel[ky, kx]
            if weight == 0:
                continue
            out += weight * gray[ky: ky + out.shape[0], kx: kx + out.shape[1]]
    return out


def laplacian_variance(rgb: np.ndarray) -> float:
    """Raw (unnormalized) Laplacian-response variance of an RGB image."""
    if rgb.shape[0] < 3 or rgb.shape[1] < 3:
        return 0.0
    gray = _to_grayscale(rgb)
    response = _convolve3x3(gray, _LAPLACIAN_KERNEL)
    return float(np.var(response))


def sharpness_score(rgb: np.ndarray) -> float:
    """Normalized sharpness in [0, 1], 1 = very sharp."""
    variance = laplacian_variance(rgb)
    return float(np.clip(np.log1p(variance) / np.log1p(_REFERENCE_VARIANCE), 0.0, 1.0))
