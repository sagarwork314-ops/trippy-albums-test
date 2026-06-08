"""Tests for the classical spectral-residual saliency map and the
"subject isolation" signal derived from it.

Pure numpy/FFT — no model, no network — so these exercise the real
algorithm end to end on synthetic fixtures built directly with numpy.
"""
from __future__ import annotations

import numpy as np

from trippy.scoring.saliency import saliency_map, subject_isolation_score


def _textured(height, width, *, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.int32).astype(np.uint8)


def _flat(height, width, *, level=128):
    return np.full((height, width, 3), level, dtype=np.uint8)


def _sharp_subject_on_plain_background(height=300, width=300, seed=1):
    """One small textured/high-detail patch, otherwise a flat, featureless frame."""
    img = _flat(height, width)
    img[100:200, 100:200] = _textured(100, 100, seed=seed)
    return img


def _blurry_patch_in_busy_scene(height=300, width=300, seed=1):
    """A statistically-different (flat) patch sitting in an otherwise busy,
    richly-textured frame -- the "odd one out" is the soft, low-detail region,
    not a sharp subject."""
    img = _textured(height, width, seed=seed + 10)
    img[100:200, 100:200] = _flat(100, 100, level=int(img[100:200, 100:200].mean()))
    return img


# --- saliency_map ---

def test_saliency_map_is_unit_range_and_matches_requested_size():
    rgb = _sharp_subject_on_plain_background()
    smap = saliency_map(rgb, size=32)
    assert smap.shape == (32, 32)
    assert smap.min() >= 0.0
    assert smap.max() <= 1.0


def test_saliency_map_is_zero_for_a_perfectly_flat_frame():
    # Nothing stands out anywhere -- no spectral residual, no saliency.
    assert np.allclose(saliency_map(_flat(200, 200)), 0.0)


# --- subject_isolation_score ---

def test_subject_isolation_score_in_unit_range():
    assert 0.0 <= subject_isolation_score(_sharp_subject_on_plain_background()) <= 1.0


def test_subject_isolation_score_rewards_a_sharp_isolated_subject_over_uniform_clutter():
    # A single in-focus subject against a clean backdrop vs. detail smeared
    # everywhere with no standout focal point.
    isolated = subject_isolation_score(_sharp_subject_on_plain_background())
    clutter = subject_isolation_score(_textured(300, 300, seed=5))
    assert isolated > clutter


def test_subject_isolation_score_rewards_focus_alignment_not_just_a_focal_point():
    # Both frames have exactly one region that "stands out" from its
    # surroundings (so saliency concentrates similarly in both) -- but in one
    # the standout region is the *sharp* one, and in the other it's the *soft*
    # one sitting amid a sharp, busy background. Only the former represents
    # "an in-focus subject set apart from its surroundings".
    sharp_subject = subject_isolation_score(_sharp_subject_on_plain_background())
    soft_spot_in_clutter = subject_isolation_score(_blurry_patch_in_busy_scene())
    assert sharp_subject > soft_spot_in_clutter
