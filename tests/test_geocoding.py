"""Tests for the optional Google Maps reverse-geocoding layer.

Per Trippy's testing philosophy, NO real network calls are made here — the
HTTP layer is fully stubbed via dependency injection (`fetch`), so these
tests stay fast, offline and deterministic, exactly like the rest of the
suite.
"""
from __future__ import annotations

import json

from trippy.scoring.geocoding import GoogleMapsGeocoder, name_places
from trippy.types import GpsPoint

SHIBUYA = GpsPoint(35.659, 139.700)
PARIS = GpsPoint(48.8566, 2.3522)


def _ok_response(formatted_address: str, types: tuple[str, ...] = ("locality",)) -> bytes:
    return json.dumps({
        "status": "OK",
        "results": [{"formatted_address": formatted_address, "types": list(types)}],
    }).encode("utf-8")


def test_geocoder_is_unavailable_without_an_api_key():
    geocoder = GoogleMapsGeocoder(None, fetch=lambda url: (_ for _ in ()).throw(AssertionError("must not call network")))
    assert geocoder.available is False
    assert geocoder.reverse_geocode(SHIBUYA) is None


def test_geocoder_returns_label_from_stubbed_response():
    geocoder = GoogleMapsGeocoder("FAKE_KEY", fetch=lambda url: _ok_response("Shibuya, Tokyo, Japan"))
    assert geocoder.available is True
    assert geocoder.reverse_geocode(SHIBUYA) == "Shibuya, Tokyo, Japan"


def test_geocoder_prefers_locality_over_street_address():
    payload = json.dumps({
        "status": "OK",
        "results": [
            {"formatted_address": "1 Chome Shibuya, Tokyo", "types": ["street_address"]},
            {"formatted_address": "Shibuya, Tokyo, Japan", "types": ["locality", "political"]},
            {"formatted_address": "Tokyo, Japan", "types": ["administrative_area_level_1"]},
        ],
    }).encode("utf-8")
    geocoder = GoogleMapsGeocoder("FAKE_KEY", fetch=lambda url: payload)
    assert geocoder.reverse_geocode(SHIBUYA) == "Shibuya, Tokyo, Japan"


def test_geocoder_caches_nearby_lookups_and_does_not_refetch():
    calls = []

    def counting_fetch(url):
        calls.append(url)
        return _ok_response("Shibuya, Tokyo, Japan")

    geocoder = GoogleMapsGeocoder("FAKE_KEY", fetch=counting_fetch)
    geocoder.reverse_geocode(GpsPoint(35.65900, 139.70000))
    geocoder.reverse_geocode(GpsPoint(35.65901, 139.70004))  # rounds to the same cache bucket

    assert len(calls) == 1


def test_geocoder_degrades_gracefully_on_network_failure():
    geocoder = GoogleMapsGeocoder("FAKE_KEY", fetch=lambda url: None)
    assert geocoder.reverse_geocode(SHIBUYA) is None


def test_geocoder_degrades_gracefully_on_api_error_status():
    payload = json.dumps({"status": "OVER_QUERY_LIMIT", "results": []}).encode("utf-8")
    geocoder = GoogleMapsGeocoder("FAKE_KEY", fetch=lambda url: payload)
    assert geocoder.reverse_geocode(SHIBUYA) is None


def test_geocoder_degrades_gracefully_on_malformed_response():
    geocoder = GoogleMapsGeocoder("FAKE_KEY", fetch=lambda url: b"not json at all")
    assert geocoder.reverse_geocode(SHIBUYA) is None


def test_name_places_makes_one_lookup_per_distinct_cluster_not_per_photo():
    places = {"a": "place-0001", "b": "place-0001", "c": "place-0001", "d": "place-0002"}
    gps_by_id = {
        "a": GpsPoint(35.6590, 139.7000),
        "b": GpsPoint(35.6591, 139.7002),
        "c": GpsPoint(35.6589, 139.7001),
        "d": PARIS,
    }
    calls = []

    def counting_fetch(url):
        calls.append(url)
        if "139.7" in url:
            return _ok_response("Shibuya, Tokyo, Japan")
        return _ok_response("Paris, France")

    geocoder = GoogleMapsGeocoder("FAKE_KEY", fetch=counting_fetch)
    labels = name_places(places, gps_by_id, geocoder)

    assert labels == {"place-0001": "Shibuya, Tokyo, Japan", "place-0002": "Paris, France"}
    # Three photos at place-0001 collapse into a single centroid lookup.
    assert len(calls) == 2


def test_name_places_returns_empty_when_geocoder_unavailable():
    places = {"a": "place-0001"}
    gps_by_id = {"a": SHIBUYA}
    geocoder = GoogleMapsGeocoder(None, fetch=lambda url: (_ for _ in ()).throw(AssertionError("must not call network")))

    assert name_places(places, gps_by_id, geocoder) == {}
