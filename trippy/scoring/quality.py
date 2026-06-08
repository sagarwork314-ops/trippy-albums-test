"""Composite "moment quality" scoring: turns raw measurements into one number
plus a human-readable explanation — answering "is this a good photo of
something worth remembering?" for a single image, in isolation.

Design notes on the composite formula
--------------------------------------
composite = printability_gate * (
    w_people * people_component +
    w_scenic * scenic_score      +
    w_appeal * appeal_score
)

`people_component` falls back to the photo's own appeal score when no faces
are detected, rather than contributing 0. Without that fallback, every
landscape/architecture shot would be structurally penalized relative to
portraits purely for lacking people — which doesn't match how a curator
actually works ("no people in this one, but it's a gorgeous shot of the
fjord" is not a worse photo for it). `scenic_score` already returns a neutral
0.5 when there's no GPS, so it needs no such fallback.

`printability_gate` is a soft multiplicative floor: a technically ruined
photo (too small to print, or hopelessly blurry) gets dragged down regardless
of how emotionally appealing the moment was — exactly the "hard-ish gate"
the curation philosophy calls for, without a harsh cliff-edge cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trippy.clustering.perceptual_hash import dhash
from trippy.scoring.color import color_cast_score, colorfulness_score
from trippy.scoring.composition import composition_score
from trippy.scoring.exposure import brightness, contrast, exposure_penalty
from trippy.scoring.eyes import OnnxEyeStateClassifier, expression_factor
from trippy.scoring.faces import OnnxFaceDetector, people_score
from trippy.scoring.resolution import resolution_score
from trippy.scoring.saliency import subject_isolation_score
from trippy.scoring.sharpness import sharpness_score
from trippy.types import PhotoFeatures, PhotoMeta, ScoreBreakdown


@dataclass(frozen=True)
class ScoringConfig:
    weight_people: float = 0.30
    weight_scenic: float = 0.20
    weight_appeal: float = 0.50

    appeal_weight_sharpness: float = 0.22
    appeal_weight_exposure: float = 0.18
    appeal_weight_contrast: float = 0.13
    appeal_weight_colorfulness: float = 0.10
    appeal_weight_color_cast: float = 0.10
    appeal_weight_composition: float = 0.12
    appeal_weight_subject_isolation: float = 0.15

    # How much weight a photo's *trip-relative percentile rank* (for
    # sharpness/colorfulness) carries against its *absolute* reference-scale
    # score. 0 = ignore the batch entirely (old behaviour); 1 = rank is all
    # that matters. A blend keeps an absolute floor (a uniformly-soft trip's
    # "best" frame still isn't called "tack sharp") while letting the
    # strongest shots of an atypical batch (overcast day, indoor-only,
    # heavy-HDR phone) stand out from their siblings more than the absolute
    # scale alone would allow.
    percentile_blend_weight: float = 0.35

    # Below this sharpness, the printability gate starts cutting hard.
    sharpness_gate_floor: float = 0.15


def extract_features(
    meta: PhotoMeta,
    rgb: np.ndarray,
    detector: OnnxFaceDetector,
    eye_classifier: OnnxEyeStateClassifier | None = None,
) -> PhotoFeatures:
    """Run every classical-CV measurement + the local face/eye-state detectors once per photo."""
    faces = detector.detect(rgb)
    eye_openness = (
        tuple(eye_classifier.eyes_open_probability(rgb, face) for face in faces)
        if eye_classifier is not None
        else tuple(None for _ in faces)
    )
    return PhotoFeatures(
        photo_id=meta.id,
        sharpness=sharpness_score(rgb),
        brightness=brightness(rgb),
        contrast=contrast(rgb),
        colorfulness=colorfulness_score(rgb),
        color_cast=color_cast_score(rgb),
        composition=composition_score(rgb),
        subject_isolation=subject_isolation_score(rgb),
        exposure_penalty=exposure_penalty(rgb),
        resolution_score=resolution_score(meta.width, meta.height),
        faces=faces,
        eye_openness=eye_openness,
        phash=dhash(rgb),
    )


def _blend_with_percentile(absolute: float, percentile: float | None, config: ScoringConfig) -> float:
    """Blends an absolute reference-scale score with a trip-relative
    percentile rank (if one was supplied — see `percentile_blend_weight`)."""
    if percentile is None:
        return absolute
    weight = config.percentile_blend_weight
    return float(np.clip((1.0 - weight) * absolute + weight * percentile, 0.0, 1.0))


def _appeal_score(
    features: PhotoFeatures,
    config: ScoringConfig,
    *,
    relative_sharpness: float | None = None,
    relative_colorfulness: float | None = None,
) -> float:
    sharpness = _blend_with_percentile(features.sharpness, relative_sharpness, config)
    colorfulness = _blend_with_percentile(features.colorfulness, relative_colorfulness, config)
    return float(np.clip(
        config.appeal_weight_sharpness * sharpness
        + config.appeal_weight_exposure * (1.0 - features.exposure_penalty)
        + config.appeal_weight_contrast * features.contrast
        + config.appeal_weight_colorfulness * colorfulness
        + config.appeal_weight_color_cast * features.color_cast
        + config.appeal_weight_composition * features.composition
        + config.appeal_weight_subject_isolation * features.subject_isolation,
        0.0, 1.0,
    ))


def _printability_gate(features: PhotoFeatures, config: ScoringConfig) -> float:
    sharpness_factor = float(np.clip(
        features.sharpness / config.sharpness_gate_floor, 0.0, 1.0
    )) if config.sharpness_gate_floor > 0 else 1.0
    # Resolution dominates (it's the harder physical limit); sharpness can
    # still drag an otherwise-high-res-but-ruined shot down.
    return float(np.clip(features.resolution_score * (0.5 + 0.5 * sharpness_factor), 0.0, 1.0))


def _build_reasons(
    features: PhotoFeatures,
    appeal: float,
    people: float,
    scenic: float,
    gate: float,
    *,
    relative_sharpness: float | None = None,
    relative_colorfulness: float | None = None,
) -> tuple[str, ...]:
    reasons: list[tuple[float, str]] = []  # (priority, text) — higher priority surfaces first

    if features.faces:
        known_eye_states = [p for p in features.eye_openness if p is not None]
        if known_eye_states and float(np.mean(known_eye_states)) < 0.4:
            reasons.append((3.5, "eyes look closed in at least one face — likely a blink frame"))
        elif people >= 0.6:
            reasons.append((3.0, f"{len(features.faces)} well-framed face(s) — strong people shot"))
        else:
            reasons.append((1.5, f"{len(features.faces)} face(s) detected, but small or off-center"))

    if scenic >= 0.7:
        reasons.append((2.5, "taken at a frequently-revisited location — likely a trip highlight"))
    elif scenic <= 0.35 and scenic > 0:
        reasons.append((0.5, "taken at a one-off spot, not a major trip focal point"))

    if features.sharpness >= 0.6:
        reasons.append((2.0, "sharp focus"))
    elif features.sharpness < 0.25:
        if relative_sharpness is not None and relative_sharpness >= 0.85:
            # Genuinely soft in absolute terms, but still the sharpest of an
            # atypical batch (overcast day, indoors, heavy phone HDR/denoise)
            # -- a curator would still pick it as "the keeper from that set".
            reasons.append((2.5, "one of the sharpest shots from this batch, even if softer overall"))
        else:
            reasons.append((4.0, "noticeably soft / blurry"))

    if features.exposure_penalty <= 0.15:
        reasons.append((1.0, "well exposed"))
    elif features.exposure_penalty >= 0.5:
        reasons.append((3.5, "poor exposure (too dark, blown out, or heavily clipped)"))

    if features.colorfulness >= 0.6:
        reasons.append((1.0, "vivid, colorful scene"))
    elif relative_colorfulness is not None and relative_colorfulness >= 0.85:
        reasons.append((1.0, "one of the most vivid shots from this batch"))

    if features.color_cast < 0.4:
        reasons.append((2.0, "noticeable color cast (white balance looks off)"))

    if features.composition >= 0.7:
        reasons.append((1.0, "well-composed (interest near the rule-of-thirds points)"))

    if features.subject_isolation >= 0.7:
        reasons.append((1.5, "clear, in-focus subject set apart from its surroundings"))
    elif features.subject_isolation < 0.3:
        reasons.append((0.5, "no clear focal point — interest is scattered or background-sharp"))

    if gate < 0.5:
        reasons.append((4.5, "limited print quality (low resolution and/or soft focus)"))

    if not reasons:
        reasons.append((0.0, "average technical quality across the board"))

    reasons.sort(key=lambda pair: -pair[0])
    return tuple(text for _, text in reasons[:4])


def score_photo(
    features: PhotoFeatures,
    scenic_score: float,
    config: ScoringConfig | None = None,
    *,
    relative_sharpness: float | None = None,
    relative_colorfulness: float | None = None,
) -> ScoreBreakdown:
    """Combine per-photo features + a (cluster-aware) scenic score into one composite.

    `relative_sharpness` / `relative_colorfulness` are optional trip-relative
    percentile ranks in [0, 1] (see `trippy/scoring/normalization.py`),
    computed once across the whole batch and passed in here — the same
    pattern as `scenic_score`. Omit them (the default) to score purely
    against the absolute reference scale, e.g. for single-photo use.
    """
    config = config or ScoringConfig()

    appeal = _appeal_score(
        features, config,
        relative_sharpness=relative_sharpness,
        relative_colorfulness=relative_colorfulness,
    )
    people = people_score(features.faces) * expression_factor(features.eye_openness)
    people_component = people if features.faces else appeal
    gate = _printability_gate(features, config)

    raw = (
        config.weight_people * people_component
        + config.weight_scenic * scenic_score
        + config.weight_appeal * appeal
    )
    composite = float(np.clip(gate * raw, 0.0, 1.0))

    reasons = _build_reasons(
        features, appeal, people, scenic_score, gate,
        relative_sharpness=relative_sharpness,
        relative_colorfulness=relative_colorfulness,
    )

    return ScoreBreakdown(
        people=people,
        scenic=scenic_score,
        appeal=appeal,
        printability=gate,
        composite=composite,
        reasons=reasons,
    )
