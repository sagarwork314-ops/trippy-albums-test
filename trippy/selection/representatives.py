"""Collapses each moment-cluster down to its best representative frame(s).

This is the "8 nearly-identical laughter shots -> 1 keeper" step. Almost
every cluster yields exactly one representative; we allow a second only when
the cluster is large *and* genuinely contains two distinct good frames worth
keeping (not just frame N and frame N+1 of the same blink), so a 30-shot
burst doesn't quietly smuggle two near-duplicates into the final set.
"""
from __future__ import annotations

from dataclasses import dataclass

from trippy.clustering.perceptual_hash import hamming_distance
from trippy.types import Moment, ScoredPhoto


@dataclass(frozen=True)
class RepresentativeConfig:
    # A cluster needs at least this many photos before a second representative
    # is even considered — small clusters are already "the best of a few".
    min_cluster_size_for_second: int = 8
    # The runner-up's score must be within this fraction of the winner's to
    # be considered "also great" rather than "clearly second-best".
    second_pick_score_ratio: float = 0.92
    # The runner-up must look meaningfully different from the winner (large
    # pHash Hamming distance) to avoid keeping two near-identical frames.
    min_phash_distance_for_second: int = 12


def select_representatives(
    moments: list[Moment],
    scored_by_id: dict[str, ScoredPhoto],
    config: RepresentativeConfig | None = None,
) -> dict[str, tuple[str, ...]]:
    """Returns moment_id -> tuple of representative photo_ids (best first)."""
    config = config or RepresentativeConfig()
    representatives: dict[str, tuple[str, ...]] = {}

    for moment in moments:
        ranked = sorted(
            moment.photo_ids,
            key=lambda pid: scored_by_id[pid].score.composite,
            reverse=True,
        )
        if not ranked:
            representatives[moment.id] = ()
            continue

        best_id = ranked[0]
        picks = [best_id]

        if len(ranked) >= config.min_cluster_size_for_second:
            best_score = scored_by_id[best_id].score.composite
            best_phash = scored_by_id[best_id].features.phash
            for candidate_id in ranked[1:]:
                candidate_score = scored_by_id[candidate_id].score.composite
                if best_score > 0 and candidate_score / best_score < config.second_pick_score_ratio:
                    break
                distance = hamming_distance(best_phash, scored_by_id[candidate_id].features.phash)
                if distance >= config.min_phash_distance_for_second:
                    picks.append(candidate_id)
                    break

        representatives[moment.id] = tuple(picks)

    return representatives
