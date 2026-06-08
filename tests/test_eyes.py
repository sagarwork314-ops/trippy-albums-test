"""Tests for the local eye-state ("eyes open?") classifier and its blend
into the people score.

Like the face detector, the bundled OCEC model is a real local ONNX model —
these tests exercise the actual model (deterministic, no network), but stay
focused on integration/plumbing rather than asserting exact "is this eye
open" judgements (synthetic PIL faces have no real eyes for it to read).
"""
from __future__ import annotations

import numpy as np

from trippy.scoring.eyes import OnnxEyeStateClassifier, _eye_crop_boxes, expression_factor
from trippy.types import FaceBox

from .fixtures import draw_face, make_rgb


def _classifier() -> OnnxEyeStateClassifier:
    return OnnxEyeStateClassifier()


def test_eye_classifier_loads_the_bundled_model():
    classifier = _classifier()
    assert classifier.available is True


def test_eye_crop_boxes_sit_within_the_face_box_symmetric_about_midline():
    face = FaceBox(x=0.2, y=0.1, width=0.4, height=0.5, confidence=0.9)
    (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2) = _eye_crop_boxes(face)

    # Both crops sit inside the vertical span of the face, in its upper half.
    for y1, y2 in ((ly1, ly2), (ry1, ry2)):
        assert face.y <= y1 < y2 <= face.y + face.height
        assert y2 < face.y + face.height * 0.6

    # Symmetric about the face's vertical midline: same width, mirrored offsets.
    midline = face.x + face.width / 2.0
    assert abs((midline - lx1) - (rx2 - midline)) < 1e-9
    assert abs((lx2 - lx1) - (rx2 - rx1)) < 1e-9
    assert lx2 < rx1  # left crop is strictly left of the right crop


def test_eyes_open_probability_returns_a_bounded_estimate_for_a_drawn_face():
    rgb = draw_face(make_rgb(seed=7), cx=0.5, cy=0.45, scale=0.4)
    face = FaceBox(x=0.3, y=0.25, width=0.4, height=0.4, confidence=0.95)

    classifier = _classifier()
    prob = classifier.eyes_open_probability(rgb, face)

    assert prob is not None
    assert 0.0 <= prob <= 1.0


def test_eyes_open_probability_is_none_for_low_confidence_faces():
    rgb = make_rgb(seed=7)
    low_confidence_face = FaceBox(x=0.3, y=0.25, width=0.3, height=0.3, confidence=0.1)

    classifier = _classifier()
    assert classifier.eyes_open_probability(rgb, low_confidence_face) is None


def test_eyes_open_probability_is_none_when_crop_falls_outside_the_frame():
    rgb = make_rgb(seed=7)
    # A face box positioned entirely outside [0, 1] -- derived eye crops can't
    # land on any real pixels.
    off_frame_face = FaceBox(x=1.5, y=1.5, width=0.3, height=0.3, confidence=0.9)

    classifier = _classifier()
    assert classifier.eyes_open_probability(rgb, off_frame_face) is None


def test_eyes_open_probability_is_none_without_a_model():
    rgb = make_rgb(seed=7)
    face = FaceBox(x=0.3, y=0.25, width=0.3, height=0.3, confidence=0.9)

    classifier = OnnxEyeStateClassifier(model_path="/nonexistent/eye_state.onnx")
    assert classifier.available is False
    assert classifier.eyes_open_probability(rgb, face) is None


# --- expression_factor (the multiplier folded into people_score) ---

def test_expression_factor_is_neutral_when_no_estimates_are_known():
    assert expression_factor(()) == 1.0
    assert expression_factor((None, None)) == 1.0


def test_expression_factor_rewards_open_eyes_and_penalizes_closed_ones():
    wide_open = expression_factor((0.95, 0.9))
    closed = expression_factor((0.05, 0.1))
    mixed = expression_factor((0.9, None))  # unknown face doesn't drag the average down

    assert wide_open > mixed > closed
    # Bounded: even a confidently-closed reading only pulls the multiplier to 0.6,
    # not zero -- a great group shot shouldn't be wrecked by one blink.
    assert 0.6 <= closed < wide_open <= 1.0


def test_expression_factor_is_monotonic_in_average_openness():
    low = expression_factor((0.2,))
    mid = expression_factor((0.5,))
    high = expression_factor((0.8,))
    assert low < mid < high
