"""Local, on-device face detection via a bundled ONNX model (no API calls).

Targets the common "UltraFace"-style architecture (e.g. version-RFB-320 /
slim-320, ~1-2MB): a single ONNX graph that takes a normalized RGB tensor
and returns per-anchor (score, box) pairs, which we filter by confidence and
de-duplicate with non-max suppression. That family of models is small enough
to vendor directly in `trippy/models/` and runs in milliseconds on CPU via
onnxruntime — fully offline, fully deterministic.

If no model file is present, `OnnxFaceDetector.available` is False and
`detect()` returns an empty tuple; the rest of the pipeline degrades
gracefully (the "people" score component simply stays neutral). See
README.md for where to obtain a compatible model file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from trippy.types import FaceBox

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detector.onnx"
_CONFIDENCE_THRESHOLD = 0.6
_NMS_IOU_THRESHOLD = 0.4

# A face filling roughly this fraction of the frame reads as "well framed"
# (a clear portrait/group shot rather than a tiny figure in the distance).
_IDEAL_FACE_AREA_FRACTION = 0.06
# Group photos are good, but we cap the head-count bonus so a crowd shot
# doesn't automatically outscore an intimate portrait.
_MAX_COUNTED_FACES = 6


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-max suppression. boxes: (N, 4) as [x1, y1, x2, y2]."""
    order = np.argsort(-scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


class OnnxFaceDetector:
    """Wraps an UltraFace-style ONNX face detector behind a tiny interface."""

    def __init__(self, model_path: Optional[Path | str] = None):
        self._session = None
        self._input_name = None
        self._input_size = (320, 240)  # (width, height); overwritten from model if known

        path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        if not path.is_file():
            return

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # errors only — silence benign graph-optimization warnings
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

    def detect(self, rgb: np.ndarray) -> tuple[FaceBox, ...]:
        """Detect faces in an (H, W, 3) uint8 RGB image.

        Returns boxes in normalized [0, 1] image coordinates so callers don't
        need to know the model's native input resolution.
        """
        if self._session is None:
            return ()

        from PIL import Image

        width, height = self._input_size
        resized = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))
        tensor = (resized.astype(np.float32) - 127.0) / 128.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]  # NCHW

        outputs = self._session.run(None, {self._input_name: tensor})
        scores, boxes = self._unpack_outputs(outputs)
        if scores is None:
            return ()

        face_scores = scores[:, 1] if scores.ndim == 2 and scores.shape[1] >= 2 else scores.reshape(-1)
        mask = face_scores > _CONFIDENCE_THRESHOLD
        if not np.any(mask):
            return ()

        kept_boxes = boxes[mask]
        kept_scores = face_scores[mask]
        keep_idx = _nms(kept_boxes, kept_scores, _NMS_IOU_THRESHOLD)

        results = []
        for idx in keep_idx:
            x1, y1, x2, y2 = (float(v) for v in kept_boxes[idx])
            x1, x2 = np.clip([x1, x2], 0.0, 1.0)
            y1, y2 = np.clip([y1, y2], 0.0, 1.0)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(FaceBox(
                x=float(x1), y=float(y1),
                width=float(x2 - x1), height=float(y2 - y1),
                confidence=float(kept_scores[idx]),
            ))
        return tuple(results)

    @staticmethod
    def _unpack_outputs(outputs: list[np.ndarray]):
        """UltraFace-style models emit (scores[1,N,2], boxes[1,N,4]) in some
        order, with boxes already in normalized [x1,y1,x2,y2] coordinates.
        We identify which output is which by its last dimension.
        """
        scores = boxes = None
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim != 2:
                continue
            if arr.shape[-1] == 4:
                boxes = arr
            elif arr.shape[-1] == 2:
                scores = _softmax(arr, axis=-1)
            elif arr.shape[-1] == 1:
                scores = arr
        return scores, boxes


def people_score(faces: tuple[FaceBox, ...]) -> float:
    """Turn detected faces into a single "people presence & framing" score in [0, 1].

    A photo with no faces scores 0 here — that's not a penalty, it just means
    this component doesn't contribute (the photo may still score well on
    scenic/appeal grounds). A photo *with* faces is rewarded for:
      - faces that are large enough to read clearly (not tiny background figures)
      - faces that are framed within the shot (not cut off at the edges)
      - having more than one face, up to a point (group shots > solo, but a
        crowd doesn't get unbounded credit over an intimate portrait)
    """
    if not faces:
        return 0.0

    per_face = []
    for face in faces:
        area_fraction = face.width * face.height
        size_score = float(np.clip(area_fraction / _IDEAL_FACE_AREA_FRACTION, 0.0, 1.0))

        cx, cy = face.x + face.width / 2.0, face.y + face.height / 2.0
        center_distance = np.hypot(cx - 0.5, cy - 0.5) / np.hypot(0.5, 0.5)
        center_score = 1.0 - float(np.clip(center_distance, 0.0, 1.0))

        # Faces cropped by the frame edge read as "badly framed" even if large/centered.
        margin = min(face.x, face.y, 1.0 - (face.x + face.width), 1.0 - (face.y + face.height))
        crop_penalty = 0.0 if margin >= 0 else float(np.clip(-margin * 4.0, 0.0, 1.0))

        per_face.append(face.confidence * size_score * (0.6 + 0.4 * center_score) * (1.0 - crop_penalty))

    average_quality = float(np.mean(per_face))
    count_bonus = (min(len(faces), _MAX_COUNTED_FACES) - 1) / (_MAX_COUNTED_FACES - 1)
    return float(np.clip(average_quality * (0.85 + 0.15 * count_bonus), 0.0, 1.0))
