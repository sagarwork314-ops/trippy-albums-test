"""EXIF parsing: timestamp + GPS extraction using exifread.

Returns plain (datetime | None, GpsPoint | None) so the rest of the
pipeline never has to know about EXIF tag names or DMS/rational encodings.
"""
from __future__ import annotations

from datetime import datetime
from fractions import Fraction
from typing import Optional

import exifread

from trippy.types import GpsPoint

_TIMESTAMP_TAGS = (
    "EXIF DateTimeOriginal",
    "Image DateTimeOriginal",
    "EXIF DateTimeDigitized",
    "Image DateTime",
)


def _parse_timestamp(tags) -> Optional[datetime]:
    for tag_name in _TIMESTAMP_TAGS:
        tag = tags.get(tag_name)
        if tag is None:
            continue
        raw = str(tag).strip()
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
    return None


def _ratio_to_float(value) -> float:
    if isinstance(value, Fraction):
        return float(value)
    return float(value.num) / float(value.den) if hasattr(value, "num") else float(value)


def _dms_to_degrees(dms_tag) -> Optional[float]:
    values = getattr(dms_tag, "values", None)
    if not values or len(values) != 3:
        return None
    degrees, minutes, seconds = (_ratio_to_float(v) for v in values)
    return degrees + minutes / 60.0 + seconds / 3600.0


def _parse_gps(tags) -> Optional[GpsPoint]:
    lat_tag = tags.get("GPS GPSLatitude")
    lat_ref_tag = tags.get("GPS GPSLatitudeRef")
    lon_tag = tags.get("GPS GPSLongitude")
    lon_ref_tag = tags.get("GPS GPSLongitudeRef")
    if lat_tag is None or lon_tag is None:
        return None

    lat = _dms_to_degrees(lat_tag)
    lon = _dms_to_degrees(lon_tag)
    if lat is None or lon is None:
        return None

    if lat_ref_tag is not None and str(lat_ref_tag).strip().upper().startswith("S"):
        lat = -lat
    if lon_ref_tag is not None and str(lon_ref_tag).strip().upper().startswith("W"):
        lon = -lon

    return GpsPoint(lat=lat, lon=lon)


def read_exif(file_obj) -> tuple[Optional[datetime], Optional[GpsPoint]]:
    """Parse timestamp + GPS from an open binary file handle.

    `file_obj` must be seekable; this function does not close it.
    """
    tags = exifread.process_file(file_obj, details=False)
    return _parse_timestamp(tags), _parse_gps(tags)
