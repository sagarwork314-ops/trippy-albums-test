"""Tests for trip-relative percentile-rank normalization and its blend into
the appeal score (sharpness/colorfulness now read relative to the rest of
the batch, not just against fixed absolute reference constants)."""
from __future__ import annotations

from trippy.scoring.normalization import percentile_ranks
from trippy.scoring.quality import ScoringConfig, score_photo
from trippy.types import FaceBox, PhotoFeatures


def _features(**overrides) -> PhotoFeatures:
    base = dict(
        photo_id="x", sharpness=0.2, brightness=0.5, contrast=0.5, colorfulness=0.6,
        color_cast=0.8, composition=0.6, subject_isolation=0.6, exposure_penalty=0.1,
        resolution_score=0.9, faces=(), eye_openness=(), phash=0,
    )
    base.update(overrides)
    return PhotoFeatures(**base)


# --- percentile_ranks ---

def test_percentile_ranks_orders_lowest_to_highest_across_unit_range():
    ranks = percentile_ranks({"a": 1.0, "b": 5.0, "c": 10.0})
    assert ranks["a"] == 0.0
    assert ranks["c"] == 1.0
    assert ranks["a"] < ranks["b"] < ranks["c"]


def test_percentile_ranks_gives_tied_values_a_shared_midpoint_rank():
    ranks = percentile_ranks({"a": 1.0, "b": 1.0, "c": 1.0, "d": 10.0})
    assert ranks["a"] == ranks["b"] == ranks["c"]
    assert ranks["a"] < ranks["d"]


def test_percentile_ranks_is_neutral_for_degenerate_batches():
    assert percentile_ranks({}) == {}
    assert percentile_ranks({"solo": 0.7}) == {"solo": 0.5}
    uniform = percentile_ranks({"a": 0.5, "b": 0.5})
    assert uniform == {"a": 0.5, "b": 0.5}


# --- blend into score_photo ---

def test_relative_rank_lifts_the_best_of_a_uniformly_soft_batch():
    # Same absolute (low) sharpness either way -- but one photo is told it's
    # the sharpest in its batch, the other that it's the softest.
    soft_features = _features(sharpness=0.2)

    standout = score_photo(soft_features, scenic_score=0.5, relative_sharpness=0.95, relative_colorfulness=0.5)
    laggard = score_photo(soft_features, scenic_score=0.5, relative_sharpness=0.05, relative_colorfulness=0.5)

    assert standout.appeal > laggard.appeal
    assert standout.composite > laggard.composite


def test_relative_rank_does_not_fully_override_the_absolute_floor():
    # A photo that's the "sharpest of a soft batch" still shouldn't out-score
    # a photo that's genuinely sharp in absolute terms (with the same relative
    # standing) -- the absolute scale remains the anchor, blended, not replaced.
    genuinely_sharp = score_photo(_features(sharpness=0.9), scenic_score=0.5, relative_sharpness=0.95)
    best_of_a_soft_batch = score_photo(_features(sharpness=0.2), scenic_score=0.5, relative_sharpness=0.95)

    assert genuinely_sharp.appeal > best_of_a_soft_batch.appeal


def test_omitting_relative_rank_falls_back_to_the_absolute_score_only():
    features = _features(sharpness=0.7, colorfulness=0.4)
    config = ScoringConfig()

    without_rank = score_photo(features, scenic_score=0.5, config=config)
    with_neutral_rank = score_photo(features, scenic_score=0.5, config=config, relative_sharpness=0.5, relative_colorfulness=0.5)

    # A neutral (0.5) percentile rank blends toward the midpoint and so is
    # *not* a no-op -- omitting the rank entirely is the true "absolute only"
    # baseline, and should differ whenever the absolute score isn't itself 0.5.
    assert without_rank.appeal != with_neutral_rank.appeal


def test_reasons_credit_a_relative_standout_in_a_soft_batch():
    breakdown = score_photo(_features(sharpness=0.15), scenic_score=0.5, relative_sharpness=0.9)
    assert any("sharpest shots from this batch" in r for r in breakdown.reasons)
