"""Shared data structures passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class GpsPoint:
    lat: float
    lon: float


@dataclass(frozen=True)
class PhotoMeta:
    """Raw metadata extracted at ingest time. One per source image file."""

    id: str
    path: str
    timestamp: Optional[datetime]
    gps: Optional[GpsPoint]
    width: int
    height: int


@dataclass(frozen=True)
class FaceBox:
    """A detected face, in normalized [0, 1] image coordinates."""

    x: float
    y: float
    width: float
    height: float
    confidence: float


@dataclass(frozen=True)
class PhotoFeatures:
    """Deterministic measurements derived from pixel data + detectors. One per photo."""

    photo_id: str
    sharpness: float
    brightness: float
    contrast: float
    colorfulness: float
    color_cast: float
    composition: float
    subject_isolation: float
    exposure_penalty: float
    resolution_score: float
    faces: tuple[FaceBox, ...]
    # Per-face P(eyes open) in [0, 1], parallel to `faces`; `None` where we
    # couldn't form an estimate (no model, low-confidence box, crop out of frame).
    eye_openness: tuple[Optional[float], ...]
    phash: int


@dataclass(frozen=True)
class ScoreBreakdown:
    """Component scores (each in [0, 1]) that combine into the composite score."""

    people: float
    scenic: float
    appeal: float
    printability: float
    composite: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScoredPhoto:
    """A photo bundled with its metadata, features and final score."""

    meta: PhotoMeta
    features: PhotoFeatures
    score: ScoreBreakdown


@dataclass(frozen=True)
class Moment:
    """A cluster of photos judged to depict the same moment (e.g. a burst)."""

    id: str
    photo_ids: tuple[str, ...]


@dataclass(frozen=True)
class CurationResult:
    """Final output of the pipeline."""

    selected: tuple[ScoredPhoto, ...]
    moments: tuple[Moment, ...]
    representatives: dict[str, tuple[str, ...]]
    all_scored: tuple[ScoredPhoto, ...]
    places: dict[str, str] = field(default_factory=dict)
    place_names: dict[str, str] = field(default_factory=dict)
