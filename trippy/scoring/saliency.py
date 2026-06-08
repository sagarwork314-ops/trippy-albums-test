"""Classical spectral-residual saliency — a deterministic, model-free read of
"where does the eye go first in this frame?", and what that implies about
whether the photo isolates a clear, in-focus subject.

Hou & Zhang's spectral residual method (2007): a natural image's log-amplitude
Fourier spectrum is dominated by a smooth, statistically predictable curve.
Subtracting a local average leaves a "residual" that spikes at statistically
*surprising* frequencies — and reconstructing the spatial domain from that
residual (keeping the original phase) lights up exactly the regions a human
eye is drawn to (faces, isolated objects, sharp edges against plain
backgrounds), while suppressing repetitive textures and flat backgrounds.
No model, no training data, no per-call cost — just an FFT on a downsampled
frame, fully deterministic and synthetic-fixture-testable.

We use the resulting saliency map for one purpose: telling apart "a photo
with a clear, in-focus subject" from "a technically sharp photo of a
cluttered or empty scene" — something a global sharpness number and a raw
edge-density grid (`composition.py`) can't distinguish on their own.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Saliency is a coarse, low-frequency property; transforming a small, fixed
# size keeps this fast and resolution-independent.
_MAP_SIZE = 64


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _box_filter(a: np.ndarray, size: int = 3) -> np.ndarray:
    """Same-shape mean filter with edge-replicated borders."""
    pad = size // 2
    padded = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a, dtype=np.float64)
    for dr in range(size):
        for dc in range(size):
            out += padded[dr:dr + a.shape[0], dc:dc + a.shape[1]]
    return out / (size * size)


def _resize(a: np.ndarray, size: int) -> np.ndarray:
    clipped = np.clip(a, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(clipped).resize((size, size), Image.Resampling.BILINEAR)).astype(np.float64)


def saliency_map(rgb: np.ndarray, *, size: int = _MAP_SIZE) -> np.ndarray:
    """Returns a `size x size` saliency map normalized to [0, 1]."""
    small = _resize(_to_grayscale(rgb), size)
    if small.std() < 1e-6:
        # A perfectly flat frame has no spectral residual to speak of -- the
        # amplitude spectrum is a single DC spike, and its log-domain
        # "residual" is pure floating-point noise. Min-max normalizing that
        # noise would manufacture fake saliency out of nothing.
        return np.zeros((size, size), dtype=np.float64)

    spectrum = np.fft.fft2(small)
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    log_amplitude = np.log(amplitude + 1e-8)
    spectral_residual = log_amplitude - _box_filter(log_amplitude, size=3)

    reconstructed = np.fft.ifft2(np.exp(spectral_residual + 1j * phase))
    saliency = np.abs(reconstructed) ** 2
    saliency = _box_filter(saliency, size=3)  # light smoothing, mirrors the paper's Gaussian step

    lo, hi = saliency.min(), saliency.max()
    if hi - lo < 1e-12:
        return np.zeros_like(saliency)
    return (saliency - lo) / (hi - lo)


def _local_detail_map(rgb: np.ndarray, *, size: int = _MAP_SIZE) -> np.ndarray:
    """Smoothed gradient-magnitude map at the same resolution as the saliency
    map — a coarse proxy for "how much fine detail/sharpness lives here"."""
    gray = _to_grayscale(rgb)
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    return _box_filter(_resize(gx + gy, size), size=3)


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float(np.clip((a * b).sum() / denom, -1.0, 1.0))


def _concentration(saliency: np.ndarray) -> float:
    """1 - normalized entropy of the saliency distribution: how much does
    visual interest gather in one place vs. spread thin across the frame?"""
    total = saliency.sum()
    if total <= 0:
        return 0.0
    p = (saliency / total).ravel()
    nonzero = p[p > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    normalized_entropy = entropy / np.log(p.size)
    return float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))


def subject_isolation_score(rgb: np.ndarray) -> float:
    """How well does this frame isolate a single, in-focus subject?

    Blends two deterministic reads of the same saliency map:

    - **focal concentration**: does visual interest gather in one place (a
      subject standing out), or spread thin across the whole frame (no
      standout — a wall of clutter, or an empty/uniform scene)?
    - **focus alignment**: does that point of interest coincide with the
      *sharpest* part of the frame? A photo where the background is crisp
      and the subject is soft (or out of frame) reads as a near-miss, not
      a keeper — something a single global sharpness number can't catch.
    """
    saliency = saliency_map(rgb)
    concentration = _concentration(saliency)

    detail = _local_detail_map(rgb)
    focus_alignment = (_normalized_correlation(saliency, detail) + 1.0) / 2.0

    return float(np.clip((concentration + focus_alignment) / 2.0, 0.0, 1.0))
