"""Optional reverse-geocoding of place clusters via the Google Maps API.

This is the one piece of Trippy that is *not* local/offline-only — by
explicit user choice, place clusters can be labelled with real place names
("Kyoto, Japan") by calling Google's Geocoding API. Everything else in the
pipeline remains deterministic and network-free; this module is opt-in
(requires an API key) and fails silently into "no name" if the key is
missing, the network is unavailable, or the API errors.

Cost/latency control: we geocode once per *place cluster* (using its
centroid), never once per photo — a 1000-photo trip might touch only a
few dozen distinct places, so this is a few dozen calls, not a thousand.
Results are cached for the lifetime of the geocoder instance.

Testability: `GoogleMapsGeocoder` takes an injectable `fetch` callable so
tests can stub the HTTP layer entirely — the test suite makes zero real
network calls, matching the rest of Trippy's offline-deterministic tests.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

from trippy.types import GpsPoint

_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"
_REQUEST_TIMEOUT_SECONDS = 5.0

# Google's reverse-geocoding response lists results from most-specific
# (street address) to least-specific (country). Street/POI-level detail is
# noisier for a "what place is this" label than neighborhood/city level, so
# we prefer mid-specificity result types when picking which result to surface.
_PREFERRED_RESULT_TYPES = (
    "point_of_interest",
    "tourist_attraction",
    "neighborhood",
    "locality",
    "administrative_area_level_2",
    "administrative_area_level_1",
    "country",
)

FetchFn = Callable[[str], Optional[bytes]]


def _default_fetch(url: str) -> Optional[bytes]:
    try:
        with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _pick_label(payload: dict) -> Optional[str]:
    results = payload.get("results") or []
    if not results:
        return None

    by_type: dict[str, str] = {}
    for result in results:
        formatted = result.get("formatted_address")
        if not formatted:
            continue
        for type_ in result.get("types", ()):
            by_type.setdefault(type_, formatted)

    for preferred in _PREFERRED_RESULT_TYPES:
        if preferred in by_type:
            return by_type[preferred]

    return results[0].get("formatted_address")


class GoogleMapsGeocoder:
    """Looks up a human-readable place name for a GPS coordinate.

    Returns `None` (never raises) on a missing key, network failure, or API
    error — reverse-geocoding is a "nice to have" label, not something that
    should ever be able to break the curation run.
    """

    def __init__(self, api_key: Optional[str], fetch: FetchFn = _default_fetch):
        self._api_key = api_key
        self._fetch = fetch
        self._cache: dict[tuple[float, float], Optional[str]] = {}

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def reverse_geocode(self, point: GpsPoint) -> Optional[str]:
        if not self._api_key:
            return None

        # Round to ~110m precision: nearby centroids collapse onto the same
        # cache entry / API call, which matters since centroids of adjacent
        # place clusters can be meters apart.
        key = (round(point.lat, 3), round(point.lon, 3))
        if key in self._cache:
            return self._cache[key]

        label = self._lookup(point)
        self._cache[key] = label
        return label

    def _lookup(self, point: GpsPoint) -> Optional[str]:
        query = urllib.parse.urlencode({
            "latlng": f"{point.lat},{point.lon}",
            "key": self._api_key,
        })
        raw = self._fetch(f"{_API_URL}?{query}")
        if raw is None:
            return None

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        if payload.get("status") != "OK":
            return None

        return _pick_label(payload)


def _centroid(points: list[GpsPoint]) -> GpsPoint:
    return GpsPoint(
        lat=sum(p.lat for p in points) / len(points),
        lon=sum(p.lon for p in points) / len(points),
    )


def name_places(
    places: dict[str, str],
    gps_by_id: dict[str, GpsPoint],
    geocoder: GoogleMapsGeocoder,
) -> dict[str, str]:
    """Returns place_id -> human-readable label for every *named* place.

    One geocoding call per distinct place cluster (at its centroid), not per
    photo — keeps API usage proportional to "places visited", not "photos
    taken". Places with no GPS, or that the geocoder can't label (no key,
    network failure, no result), are simply absent from the returned dict;
    callers should fall back to the existing place-id-based labels.
    """
    if not geocoder.available:
        return {}

    points_by_place: dict[str, list[GpsPoint]] = {}
    for photo_id, place_id in places.items():
        gps = gps_by_id.get(photo_id)
        if gps is not None:
            points_by_place.setdefault(place_id, []).append(gps)

    labels: dict[str, str] = {}
    for place_id, points in points_by_place.items():
        label = geocoder.reverse_geocode(_centroid(points))
        if label:
            labels[place_id] = label

    return labels
