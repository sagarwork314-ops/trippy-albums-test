"""Synthetic fixture generators — no real photos or network calls needed.

Every test in this suite builds its inputs programmatically with controllable
"knobs" (sharpness, exposure, color, faces, EXIF timestamp/GPS), so behaviour
is deterministic and the suite runs anywhere, instantly, offline.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import piexif
from PIL import Image, ImageDraw

from trippy.types import GpsPoint, PhotoMeta

_RNG = np.random.default_rng(1234)


def make_rgb(
    height: int = 600,
    width: int = 800,
    *,
    sharp: bool = True,
    brightness: float = 0.5,
    colorful: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Build a synthetic RGB array with controllable quality knobs.

    - sharp=False downsamples-then-upsamples to produce genuine blur (the
      same operation a Laplacian-variance metric is designed to catch).
    - brightness shifts the mid-point of the pixel value distribution.
    - colorful=False produces a near-grayscale image (low saturation).
    """
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    base_level = int(np.clip(brightness, 0.0, 1.0) * 255)
    spread = 90

    if colorful:
        rgb = rng.integers(
            max(base_level - spread, 0), min(base_level + spread, 256), (height, width, 3), dtype=np.int32
        ).astype(np.uint8)
    else:
        gray = rng.integers(
            max(base_level - spread, 0), min(base_level + spread, 256), (height, width), dtype=np.int32
        ).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

    if not sharp:
        small = Image.fromarray(rgb).resize((max(width // 16, 1), max(height // 16, 1)))
        rgb = np.asarray(small.resize((width, height), Image.Resampling.BILINEAR))

    return rgb


def draw_face(rgb: np.ndarray, *, cx: float = 0.5, cy: float = 0.5, scale: float = 0.3) -> np.ndarray:
    """Draws a crude oval "face" with two eye blobs at a normalized position/size.

    Crude is fine — it only needs to be face-*shaped* enough for a small
    general-purpose face detector to register a hit; the test suite checks
    relative behaviour (faces detected vs. not), not exact box coordinates.
    """
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)

    face_w, face_h = scale * w, scale * 1.3 * h
    x0, y0 = cx * w - face_w / 2, cy * h - face_h / 2
    x1, y1 = x0 + face_w, y0 + face_h
    draw.ellipse([x0, y0, x1, y1], fill=(225, 195, 175))

    eye_w, eye_h = face_w * 0.18, face_h * 0.1
    eye_y = y0 + face_h * 0.35
    draw.ellipse([x0 + face_w * 0.22, eye_y, x0 + face_w * 0.22 + eye_w, eye_y + eye_h], fill=(40, 30, 25))
    draw.ellipse([x1 - face_w * 0.22 - eye_w, eye_y, x1 - face_w * 0.22, eye_y + eye_h], fill=(40, 30, 25))
    draw.arc([x0 + face_w * 0.25, y0 + face_h * 0.55, x1 - face_w * 0.25, y0 + face_h * 0.8],
             start=20, end=160, fill=(140, 70, 70), width=max(int(face_w * 0.03), 1))

    return np.asarray(img)


def make_meta(
    photo_id: str,
    *,
    path: str = "synthetic.jpg",
    timestamp: datetime | None = None,
    gps: GpsPoint | None = None,
    width: int = 800,
    height: int = 600,
) -> PhotoMeta:
    return PhotoMeta(id=photo_id, path=path, timestamp=timestamp, gps=gps, width=width, height=height)


def _deg_to_dms_rational(deg: float) -> list[tuple[int, int]]:
    deg = abs(deg)
    d = int(deg)
    m_float = (deg - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    return [(d, 1), (m, 1), (int(round(s * 100)), 100)]


def write_jpeg_with_exif(
    path: Path,
    rgb: np.ndarray,
    *,
    timestamp: datetime | None = None,
    gps: GpsPoint | None = None,
) -> Path:
    """Writes an RGB array to disk as a JPEG carrying EXIF timestamp/GPS —
    used for pipeline-level (folder-scanning) integration tests."""
    img = Image.fromarray(rgb)

    exif_dict: dict = {"0th": {}, "Exif": {}, "GPS": {}}
    if timestamp is not None:
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = timestamp.strftime("%Y:%m:%d %H:%M:%S").encode()
    if gps is not None:
        exif_dict["GPS"] = {
            piexif.GPSIFD.GPSLatitudeRef: b"N" if gps.lat >= 0 else b"S",
            piexif.GPSIFD.GPSLatitude: _deg_to_dms_rational(gps.lat),
            piexif.GPSIFD.GPSLongitudeRef: b"E" if gps.lon >= 0 else b"W",
            piexif.GPSIFD.GPSLongitude: _deg_to_dms_rational(gps.lon),
        }

    exif_bytes = piexif.dump(exif_dict)
    img.save(path, "jpeg", exif=exif_bytes, quality=92)
    return path
