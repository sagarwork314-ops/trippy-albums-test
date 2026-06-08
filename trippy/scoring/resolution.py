"""Print-worthiness from raw resolution: a hard-ish technical floor.

No amount of compositional or emotional value rescues a photo that's too
small to print decently. We score effective megapixels against the
resolution needed for a respectable print, and treat extreme aspect ratios
(panoramas, screenshots) as a mild penalty since they don't fit standard
print formats well.
"""
from __future__ import annotations

import numpy as np

# Megapixels needed for an acceptable small print (~4x6 @ 200 dpi) and a
# comfortably large one (~8x10 @ 300 dpi). Linearly interpolate between them.
_MIN_ACCEPTABLE_MP = 0.7
_GOOD_MP = 6.0

_IDEAL_ASPECT = 3.0 / 2.0  # classic photo-print ratio
_ASPECT_TOLERANCE = 1.2  # ratios within +/- this multiple incur no penalty


def resolution_score(width: int, height: int) -> float:
    megapixels = (width * height) / 1_000_000.0
    res_factor = np.clip(
        (megapixels - _MIN_ACCEPTABLE_MP) / (_GOOD_MP - _MIN_ACCEPTABLE_MP), 0.0, 1.0
    )

    long_side, short_side = max(width, height), max(min(width, height), 1)
    ratio = long_side / short_side
    excess = max(0.0, ratio - _IDEAL_ASPECT * _ASPECT_TOLERANCE)
    aspect_factor = np.clip(1.0 - excess, 0.0, 1.0)

    return float(res_factor * (0.85 + 0.15 * aspect_factor))
