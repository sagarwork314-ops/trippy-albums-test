"""GPS distance helpers."""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from trippy.types import GpsPoint

_EARTH_RADIUS_M = 6_371_000.0


def haversine_meters(a: GpsPoint, b: GpsPoint) -> float:
    """Great-circle distance between two GPS points, in meters."""
    lat1, lon1, lat2, lon2 = (radians(v) for v in (a.lat, a.lon, b.lat, b.lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * asin(sqrt(h))
