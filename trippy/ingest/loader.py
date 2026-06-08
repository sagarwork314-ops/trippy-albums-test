"""Scans a folder of images and produces PhotoMeta + decoded pixel arrays.

Kept deliberately dumb: no scoring or clustering happens here. This is the
single place that touches the filesystem and image codecs, so the rest of
the pipeline can stay pure and easy to test with synthetic data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from trippy.ingest.exif import read_exif
from trippy.types import PhotoMeta

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"})


@dataclass(frozen=True)
class LoadedPhoto:
    """A decoded photo: metadata plus an RGB pixel array (H, W, 3) uint8."""

    meta: PhotoMeta
    rgb: np.ndarray


def _make_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def iter_image_paths(folder: str | Path) -> list[Path]:
    root = Path(folder)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_photo(path: str | Path) -> LoadedPhoto:
    path = Path(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        rgb = np.asarray(img)

    timestamp, gps = None, None
    try:
        with open(path, "rb") as fh:
            timestamp, gps = read_exif(fh)
    except OSError:
        pass

    height, width = rgb.shape[:2]
    meta = PhotoMeta(
        id=_make_id(path),
        path=str(path),
        timestamp=timestamp,
        gps=gps,
        width=width,
        height=height,
    )
    return LoadedPhoto(meta=meta, rgb=rgb)


def load_folder(folder: str | Path) -> list[LoadedPhoto]:
    return [load_photo(p) for p in iter_image_paths(folder)]
