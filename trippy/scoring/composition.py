"""Composition score from a 3x3 edge-density grid.

Two complementary, deterministic signals — no learned aesthetic model:

1. **Rule-of-thirds alignment**: the four "power points" sit at the inner
   corners of the center cell of a 3x3 grid (touching the four corner
   cells). Photos that concentrate their visual interest near those points,
   rather than dead-center or uniformly everywhere, read as more
   deliberately composed.

2. **Energy concentration**: alignment alone is fooled by busy/cluttered
   backgrounds — a chaotic frame can easily dump 55%+ of its (very large)
   edge energy into the corner cells by sheer volume. We counter this with
   the *concentration* of the energy distribution across all nine cells
   (1 - normalized entropy): a deliberate composition tends to draw the eye
   to a fairly small part of the frame (subject against comparatively clean
   surroundings), while clutter spreads detail evenly across the whole
   grid. High edge energy that is *evenly smeared* over all nine cells now
   reads as noise rather than as "great composition".

The final score blends both — alignment matters more, but a photo only
reaches the top of the range when it also reads as having a clear subject
rather than uniform visual chaos.
"""
from __future__ import annotations

import numpy as np

# A perfectly uniform image would put 4/9 of its edge energy in the four
# corner cells; we treat moderately *more* than that as well-aligned and
# scale up to this target before clamping to [0, 1].
_TARGET_CORNER_FRACTION = 0.55

# Normalized entropy (0 = all energy in one cell, 1 = perfectly uniform
# across all nine) above which we treat the frame as "uniformly busy" —
# i.e. no one part of it stands out as the subject.
_MAX_WELL_CONCENTRATED_ENTROPY = 0.97

_ALIGNMENT_WEIGHT = 0.6
_CONCENTRATION_WEIGHT = 0.4


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    return gx + gy


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _cell_energy(edges: np.ndarray) -> np.ndarray:
    h, w = edges.shape
    row_bounds = (0, h // 3, 2 * h // 3, h)
    col_bounds = (0, w // 3, 2 * w // 3, w)

    cell_energy = np.zeros((3, 3), dtype=np.float64)
    for r in range(3):
        for c in range(3):
            cell = edges[row_bounds[r]:row_bounds[r + 1], col_bounds[c]:col_bounds[c + 1]]
            cell_energy[r, c] = cell.sum()
    return cell_energy


def _alignment_score(proportions: np.ndarray) -> float:
    corner_fraction = (
        proportions[0, 0] + proportions[0, 2] + proportions[2, 0] + proportions[2, 2]
    )
    return float(np.clip(corner_fraction / _TARGET_CORNER_FRACTION, 0.0, 1.0))


def _concentration_score(proportions: np.ndarray) -> float:
    p = proportions.ravel()
    nonzero = p[p > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    normalized_entropy = entropy / np.log(p.size)
    concentration = 1.0 - normalized_entropy
    return float(np.clip(concentration / (1.0 - _MAX_WELL_CONCENTRATED_ENTROPY), 0.0, 1.0))


def composition_score(rgb: np.ndarray) -> float:
    gray = _to_grayscale(rgb)
    edges = _gradient_magnitude(gray)
    cell_energy = _cell_energy(edges)

    total = cell_energy.sum()
    if total <= 0:
        return 0.0

    proportions = cell_energy / total
    alignment = _alignment_score(proportions)
    concentration = _concentration_score(proportions)
    return float(np.clip(_ALIGNMENT_WEIGHT * alignment + _CONCENTRATION_WEIGHT * concentration, 0.0, 1.0))
