"""Local, on-device "eyes open?" scoring — the single most common reason a
human curator rejects an otherwise-good people shot ("great shot, but she
blinked"), and a gap the rest of the pipeline can't see.

We bundle OCEC (github.com/PINTO0309/OCEC, MIT licensed), a tiny (~112KB)
binary eye-state classifier that takes a small RGB eye crop and returns
P(eyes open) via sigmoid. It expects a *tightly cropped single eye*, and we
have no eye-landmark detector — only the face bounding box from
`OnnxFaceDetector`. So we derive approximate left/right eye crop regions from
standard frontal-face proportions (eyes sit roughly 30-40% down from the top
of the box, symmetric about the vertical midline).

This is a heuristic, not a precise localization — it degrades gracefully for
profile shots or extreme head tilts (the crop may miss the eye, in which case
the classifier just sees skin/hair and returns a noisy-but-bounded estimate,
averaged away across both eyes and, usually, multiple faces). For the common
case this pipeline cares about — frontal portraits and group shots — it's a
deterministic, fully local signal that closed-eye "blink" frames lack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from trippy.types import FaceBox

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "eye_state.onnx"

# Empirical frontal-face proportions (fractions of the face box) for where
# the eyes sit — roughly the upper-middle third, symmetric about the midline.
_EYE_CENTER_Y_FRACTION = 0.38
_LEFT_EYE_CENTER_X_FRACTION = 0.30
_RIGHT_EYE_CENTER_X_FRACTION = 0.70
_EYE_CROP_WIDTH_FRACTION = 0.24
_EYE_CROP_HEIGHT_FRACTION = 0.18

# Below this confidence we don't trust the face box enough to bother deriving
# eye crops from it — garbage-in/garbage-out for a heuristic localization.
_MIN_FACE_CONFIDENCE = 0.5


def _eye_crop_boxes(face: FaceBox) -> tuple[tuple[float, float, float, float], ...]:
    """Returns (x1, y1, x2, y2) in normalized [0, 1] image coords for each eye."""
    cy = face.y + _EYE_CENTER_Y_FRACTION * face.height
    half_w = (_EYE_CROP_WIDTH_FRACTION * face.width) / 2.0
    half_h = (_EYE_CROP_HEIGHT_FRACTION * face.height) / 2.0

    boxes = []
    for x_fraction in (_LEFT_EYE_CENTER_X_FRACTION, _RIGHT_EYE_CENTER_X_FRACTION):
        cx = face.x + x_fraction * face.width
        boxes.append((cx - half_w, cy - half_h, cx + half_w, cy + half_h))
    return tuple(boxes)


class OnnxEyeStateClassifier:
    """Wraps the bundled OCEC ONNX eye-state classifier behind a tiny interface."""

    def __init__(self, model_path: Optional[Path | str] = None):
        self._session = None
        self._input_name = None
        self._input_size = (40, 24)  # (width, height)

        path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        if not path.is_file():
            return

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(str(path), sess_options=sess_options, providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        shape = input_meta.shape
        if isinstance(shape[3], int) and isinstance(shape[2], int):
            self._input_size = (shape[3], shape[2])  # (W, H) from (N, C, H, W)

        self._session = session
        self._input_name = input_meta.name

    @property
    def available(self) -> bool:
        return self._session is not None

    def _classify_crop(self, crop: np.ndarray) -> Optional[float]:
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None

        from PIL import Image

        width, height = self._input_size
        resized = np.asarray(Image.fromarray(crop).resize((width, height), Image.Resampling.BILINEAR))
        tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        outputs = self._session.run(None, {self._input_name: tensor})
        return float(np.clip(np.squeeze(outputs[0]), 0.0, 1.0))

    def eyes_open_probability(self, rgb: np.ndarray, face: FaceBox) -> Optional[float]:
        """Estimates P(eyes open) for one detected face, or `None` if the
        face is too low-confidence to bother, or both derived eye crops fall
        outside the image / are degenerate.

        Combines both eyes with `min`, not `mean`: a portrait where one eye
        is clearly closed (a mid-blink frame) should read as "eyes not open"
        even if the other eye looks fine — that's how a human curator judges it.
        """
        if self._session is None or face.confidence < _MIN_FACE_CONFIDENCE:
            return None

        h, w = rgb.shape[0], rgb.shape[1]
        probs = []
        for x1, y1, x2, y2 in _eye_crop_boxes(face):
            px1, px2 = int(np.clip(x1, 0, 1) * w), int(np.clip(x2, 0, 1) * w)
            py1, py2 = int(np.clip(y1, 0, 1) * h), int(np.clip(y2, 0, 1) * h)
            if px2 <= px1 or py2 <= py1:
                continue
            prob = self._classify_crop(rgb[py1:py2, px1:px2])
            if prob is not None:
                probs.append(prob)

        if not probs:
            return None
        return float(min(probs))


def expression_factor(eye_openness: tuple[Optional[float], ...]) -> float:
    """Turns per-face eye-openness estimates into a single multiplier in
    [0.6, 1.0] applied to the people score.

    Faces we couldn't estimate (no model, low-confidence detection, crop out
    of frame) simply don't participate — we'd rather stay neutral than
    penalize a photo for a signal we don't trust. Only photos where we *do*
    have a confident "eyes look closed" reading get pulled down; this avoids
    double-penalizing things sharpness/exposure already catch (motion blur
    during a blink tends to also tank the sharpness score).
    """
    known = [p for p in eye_openness if p is not None]
    if not known:
        return 1.0

    average_open = float(np.mean(known))
    # Map [0, 1] openness -> [0.6, 1.0] multiplier: even a confidently-closed
    # reading only pulls the people component down by 40%, not to zero --
    # plenty of great candid/laughing shots have a mid-blink in the group.
    return float(np.clip(0.6 + 0.4 * average_open, 0.6, 1.0))
