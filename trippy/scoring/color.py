"""Color metrics: colorfulness (Hasler & Süsstrunk, 2003) and color-cast
detection — both reference-free, classical, and deterministic.
"""
from __future__ import annotations

import numpy as np

# Empirical colorfulness values above which an image reads as "vivid" to
# most viewers; used purely to rescale the raw metric into [0, 1].
_REFERENCE_COLORFULNESS = 90.0


def colorfulness_score(rgb: np.ndarray) -> float:
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)

    rg = r - g
    yb = 0.5 * (r + g) - b

    std_root = np.sqrt(np.var(rg) + np.var(yb))
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    raw = std_root + 0.3 * mean_root

    return float(np.clip(raw / _REFERENCE_COLORFULNESS, 0.0, 1.0))


# "White patch" / Retinex assumption: the brightest non-clipped pixels in a
# typical photo are usually specular highlights or near-white surfaces
# (sky, clouds, paper, eyes, light walls) -- regions that take on the color
# of the *light source* almost regardless of the object's true color. Their
# average color is therefore a good estimate of the illuminant: neutral
# means well-balanced, tinted means a cast. We sample this top slice...
_BRIGHT_SAMPLE_FRACTION = 0.05
# ...excluding pixels so bright they've clipped, since clipping erases the
# very color information we need.
_CLIPPED_LUMINANCE = 250.0

# Need at least this many non-clipped pixels before any of this is meaningful.
_MIN_SAMPLE_PIXEL_FRACTION = 0.02

# If the sampled "brightest" pixels are *themselves* strongly saturated, the
# frame has no near-white reference at all -- its brightest spots are simply
# vividly colored on purpose (an orange sunset, a neon sign), not evidence of
# a white-balance miss. Above this saturation we don't trust the sample.
_MAX_TRUSTED_SAMPLE_SATURATION = 40.0

# Chroma deviation from neutral (0-255 scale) at/above which we treat the
# cast as maximally severe; used to rescale into [0, 1].
_MAX_CAST_MAGNITUDE = 40.0


def color_cast_score(rgb: np.ndarray) -> float:
    """1.0 = neutral / well white-balanced, 0.0 = strong unwanted color cast.

    Estimates the light source's color from the brightest non-clipped pixels
    (the "white patch" / Retinex assumption -- highlights and near-white
    surfaces take on the illuminant's tint almost regardless of the object's
    true color) and checks whether that estimate drifts from neutral gray.

    We deliberately *don't* try to find "low-saturation pixels" directly --
    a strong cast itself raises the apparent saturation of what should be
    neutral content, which would make that approach blind to exactly the
    thing it's trying to measure. Sampling by brightness sidesteps that.

    To avoid false positives on scenes that are vividly colored *on purpose*
    (a sunset, a neon-lit street, a lush forest canopy -- where even the
    brightest spots are saturated), we only trust the sample when it itself
    looks like it was *trying* to be near-white; otherwise we stay neutral.
    """
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)

    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    non_clipped = luminance < _CLIPPED_LUMINANCE
    if non_clipped.mean() < _MIN_SAMPLE_PIXEL_FRACTION:
        return 1.0

    bright_threshold = np.quantile(luminance[non_clipped], 1.0 - _BRIGHT_SAMPLE_FRACTION)
    sample = non_clipped & (luminance >= bright_threshold)
    rs, gs, bs = r[sample], g[sample], b[sample]

    sample_saturation = np.maximum(np.maximum(rs, gs), bs) - np.minimum(np.minimum(rs, gs), bs)
    if sample_saturation.mean() > _MAX_TRUSTED_SAMPLE_SATURATION:
        return 1.0

    illuminant_estimate = np.array([rs.mean(), gs.mean(), bs.mean()])
    chroma = illuminant_estimate / illuminant_estimate.mean()
    cast_magnitude = float(np.std(chroma) * 255.0)
    return float(np.clip(1.0 - cast_magnitude / _MAX_CAST_MAGNITUDE, 0.0, 1.0))
