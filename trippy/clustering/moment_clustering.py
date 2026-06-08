"""Groups photos into "moments": bursts and near-duplicates collapse into one.

A human curator doesn't see 8 nearly-identical shots of the same laugh as
8 separate memories — they see *one* moment and pick its best frame. We
approximate that judgement deterministically by connecting two photos into
the same moment whenever EITHER:

  1. They were taken close together in time AND close together in space
     (classic burst / "same stop on the trip" signal), OR
  2. They look visually near-identical (dHash Hamming distance below a
     threshold) — catches bursts even when GPS is missing or noisy, and
     catches near-dupes shot a few minutes apart.

Connectivity is transitive (union-find), so a 30-shot burst collapses into
one cluster even though the first and last frame might individually look
quite different from each other.

Photos are compared only within a sliding time window (not all-pairs), which
keeps this O(n * window) instead of O(n^2) — moments are local in time, so
nothing meaningful is lost.
"""
from __future__ import annotations

from dataclasses import dataclass

from trippy.clustering.geo import haversine_meters
from trippy.clustering.perceptual_hash import hamming_distance
from trippy.clustering.union_find import UnionFind
from trippy.types import Moment, PhotoFeatures, PhotoMeta


@dataclass(frozen=True)
class ClusteringConfig:
    time_threshold_seconds: float = 120.0
    gps_threshold_meters: float = 100.0
    phash_threshold: int = 10
    # How many subsequent (time-sorted) photos to compare each photo against.
    # Bursts rarely exceed a few dozen frames; this bounds the search without
    # missing realistic same-moment pairs.
    max_window: int = 40


def _is_same_moment(
    a: PhotoMeta, a_feat: PhotoFeatures,
    b: PhotoMeta, b_feat: PhotoFeatures,
    config: ClusteringConfig,
) -> bool:
    if a.timestamp is not None and b.timestamp is not None:
        gap = abs((b.timestamp - a.timestamp).total_seconds())
        if gap <= config.time_threshold_seconds:
            if a.gps is not None and b.gps is not None:
                if haversine_meters(a.gps, b.gps) <= config.gps_threshold_meters:
                    return True
            else:
                # No GPS to corroborate or contradict — time-proximity alone
                # is a reasonable burst signal over short windows.
                return True

    if hamming_distance(a_feat.phash, b_feat.phash) <= config.phash_threshold:
        return True

    return False


def _sort_key(meta: PhotoMeta, fallback_index: int):
    # Photos without timestamps sort after timestamped ones, in their
    # original (e.g. filename) order, so they still get compared to their
    # filesystem-neighbors rather than scattered randomly.
    if meta.timestamp is not None:
        return (0, meta.timestamp.timestamp(), fallback_index)
    return (1, 0.0, fallback_index)


def cluster_moments(
    metas: list[PhotoMeta],
    features: dict[str, PhotoFeatures],
    config: ClusteringConfig | None = None,
) -> list[Moment]:
    """Cluster photos into moments. Returns one Moment per connected component,
    in chronological order of its earliest photo.
    """
    config = config or ClusteringConfig()
    if not metas:
        return []

    order = sorted(range(len(metas)), key=lambda i: _sort_key(metas[i], i))
    uf = UnionFind(len(metas))

    for window_pos, i in enumerate(order):
        meta_i = metas[i]
        feat_i = features[meta_i.id]
        for j in order[window_pos + 1: window_pos + 1 + config.max_window]:
            meta_j = metas[j]
            # Time-sorted window: once two timestamped photos exceed the time
            # threshold, every later photo in the window will too — stop early.
            if meta_i.timestamp is not None and meta_j.timestamp is not None:
                gap = abs((meta_j.timestamp - meta_i.timestamp).total_seconds())
                if gap > config.time_threshold_seconds * config.max_window:
                    break
            feat_j = features[meta_j.id]
            if _is_same_moment(meta_i, feat_i, meta_j, feat_j, config):
                uf.union(i, j)

    groups = uf.groups()

    def earliest_timestamp(group: list[int]):
        timestamps = [metas[idx].timestamp for idx in group if metas[idx].timestamp is not None]
        if timestamps:
            return (0, min(t.timestamp() for t in timestamps))
        return (1, min(group))

    groups.sort(key=earliest_timestamp)

    moments = []
    for group_index, group in enumerate(groups):
        ordered_ids = tuple(metas[idx].id for idx in sorted(group, key=lambda idx: _sort_key(metas[idx], idx)))
        moments.append(Moment(id=f"moment-{group_index:04d}", photo_ids=ordered_ids))
    return moments
