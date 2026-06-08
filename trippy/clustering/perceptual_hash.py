"""Difference-hash (dHash) perceptual hashing for near-duplicate / burst detection.

dHash is robust to minor exposure/crop differences between consecutive burst
shots while still being a single 64-bit integer that's cheap to compare via
Hamming distance — exactly what we need to tell "same moment, different frame"
apart from "different moment".
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_HASH_SIZE = 8  # -> 8x8 horizontal-difference bits = 64-bit hash


def dhash(rgb: np.ndarray) -> int:
    """Compute a 64-bit difference hash from an (H, W, 3) uint8 RGB array."""
    img = Image.fromarray(rgb).convert("L").resize(
        (_HASH_SIZE + 1, _HASH_SIZE), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(img, dtype=np.int16)

    diff = pixels[:, :-1] > pixels[:, 1:]
    bits = diff.flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
