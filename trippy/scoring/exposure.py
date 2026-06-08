"""Brightness, contrast and clipping-based exposure quality.

These three numbers, taken together, catch the most common technical
failures a curator would reject on sight: photos that are too dark, blown
out, flat/hazy, or harshly clipped at either end of the histogram.
"""
from __future__ import annotations

import numpy as np

_IDEAL_BRIGHTNESS = 0.5  # mid-gray; deviation from this is penalized
_CLIP_THRESHOLD_LOW = 8  # 0-255 scale
_CLIP_THRESHOLD_HIGH = 247


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def brightness(rgb: np.ndarray) -> float:
    """Mean luminance, normalized to [0, 1]."""
    return float(np.mean(_to_grayscale(rgb)) / 255.0)


def contrast(rgb: np.ndarray) -> float:
    """Luminance std-dev, normalized to [0, 1] (1 == maximal spread)."""
    return float(np.std(_to_grayscale(rgb)) / 127.5)


def clipping_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels that are crushed black or blown-out white."""
    gray = _to_grayscale(rgb)
    clipped = (gray <= _CLIP_THRESHOLD_LOW) | (gray >= _CLIP_THRESHOLD_HIGH)
    return float(np.mean(clipped))


def exposure_penalty(rgb: np.ndarray) -> float:
    """Combined exposure penalty in [0, 1]; 0 = well-exposed, 1 = unusable.

    Blends "how far is mean brightness from mid-gray" with "how much of the
    frame is clipped" — either alone can be misleading (a beach scene is
    legitimately bright; a night scene is legitimately dark), but a photo
    that is both far from mid-gray *and* heavily clipped is reliably bad.
    """
    b = brightness(rgb)
    deviation = abs(b - _IDEAL_BRIGHTNESS) / _IDEAL_BRIGHTNESS  # in [0, ~1]
    clip = clipping_fraction(rgb)
    penalty = 0.5 * np.clip(deviation, 0.0, 1.0) + 0.5 * np.clip(clip * 4.0, 0.0, 1.0)
    return float(np.clip(penalty, 0.0, 1.0))
