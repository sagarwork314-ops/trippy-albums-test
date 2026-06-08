"""Scenic / landmark significance, derived purely from GPS convergence.

We deliberately avoid any reverse-geocoding API: instead we lean on a simple
but powerful observation a human curator makes implicitly — "we kept coming
back to this spot" or "lots of separate moments happened around here" is a
strong, deterministic signal that a place mattered on the trip (a famous
viewpoint, the hotel's beach, the old town square), versus a single quick
snapshot taken in transit.

Concretely: cluster all GPS-tagged photos into "places" by spatial proximity,
then count how many *distinct moments* (not raw photos — a 40-shot burst
shouldn't outweigh four separate five-minute stops) touched each place.
Places touched by more distinct moments score higher. Photos without GPS get
a neutral score, since absence of location data is not evidence of dullness.
"""
from __future__ import annotations

from dataclasses import dataclass

from trippy.clustering.geo import haversine_meters
from trippy.clustering.union_find import UnionFind
from trippy.types import GpsPoint, Moment, PhotoMeta

_NEUTRAL_SCORE = 0.5
_BASELINE_PLACE_SCORE = 0.3  # any real, GPS-tagged place gets at least this


@dataclass(frozen=True)
class ScenicConfig:
    place_radius_meters: float = 300.0


def _cluster_places(
    photo_ids_with_gps: list[str], gps_by_id: dict[str, GpsPoint], radius: float
) -> list[list[str]]:
    n = len(photo_ids_with_gps)
    uf = UnionFind(n)
    for i in range(n):
        gps_i = gps_by_id[photo_ids_with_gps[i]]
        for j in range(i + 1, n):
            if haversine_meters(gps_i, gps_by_id[photo_ids_with_gps[j]]) <= radius:
                uf.union(i, j)

    clusters: dict[int, list[str]] = {}
    for i, photo_id in enumerate(photo_ids_with_gps):
        clusters.setdefault(uf.find(i), []).append(photo_id)
    return list(clusters.values())


def assign_places(metas: list[PhotoMeta], config: ScenicConfig | None = None) -> dict[str, str]:
    """Returns photo_id -> place_id for every photo.

    GPS-tagged photos are grouped into spatial clusters ("place-0001", ...).
    Photos without GPS each get a unique singleton place id — we have no
    location signal to group them by, so treating them as one big bucket
    would (wrongly) imply they're all "the same place" for diversity purposes.
    """
    config = config or ScenicConfig()

    gps_by_id = {m.id: m.gps for m in metas if m.gps is not None}
    photo_ids_with_gps = list(gps_by_id.keys())

    place_by_photo: dict[str, str] = {}
    if photo_ids_with_gps:
        for cluster_index, cluster in enumerate(
            _cluster_places(photo_ids_with_gps, gps_by_id, config.place_radius_meters)
        ):
            for photo_id in cluster:
                place_by_photo[photo_id] = f"place-{cluster_index:04d}"

    for meta in metas:
        if meta.id not in place_by_photo:
            place_by_photo[meta.id] = f"place-unknown-{meta.id}"

    return place_by_photo


def compute_scenic_scores(
    metas: list[PhotoMeta],
    moments: list[Moment],
    config: ScenicConfig | None = None,
    places: dict[str, str] | None = None,
) -> dict[str, float]:
    """Returns photo_id -> scenic score in [0, 1].

    `places` may be passed in (e.g. from `assign_places`) to avoid
    re-clustering GPS points when the caller already needs that grouping for
    diversity selection too.
    """
    config = config or ScenicConfig()

    photo_to_moment = {
        photo_id: moment.id for moment in moments for photo_id in moment.photo_ids
    }

    gps_ids = {m.id for m in metas if m.gps is not None}
    scores: dict[str, float] = {m.id: _NEUTRAL_SCORE for m in metas if m.gps is None}
    if not gps_ids:
        return scores

    places = places if places is not None else assign_places(metas, config)

    clusters: dict[str, list[str]] = {}
    for photo_id in gps_ids:
        clusters.setdefault(places[photo_id], []).append(photo_id)

    notability_by_place = {
        place_id: len({photo_to_moment[pid] for pid in photo_ids})
        for place_id, photo_ids in clusters.items()
    }
    max_notability = max(notability_by_place.values()) if notability_by_place else 1

    for place_id, photo_ids in clusters.items():
        notability = notability_by_place[place_id]
        if max_notability <= 1:
            place_score = _NEUTRAL_SCORE
        else:
            fraction = (notability - 1) / (max_notability - 1)
            place_score = _BASELINE_PLACE_SCORE + (1.0 - _BASELINE_PLACE_SCORE) * fraction
        for photo_id in photo_ids:
            scores[photo_id] = place_score

    return scores
