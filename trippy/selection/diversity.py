"""Greedy day/place-quota selection: picks the final ~N so the result tells
the story of the *whole* trip, not just its most-photographed afternoon.

A human curator doing this by hand naturally thinks "okay, we need at least
one or two from each day, and within each day we should cover the different
places we went" — then fills any remaining slots with whatever was best
overall. That's exactly the three-pass structure below:

  1. Day quotas      — split the target count across days, weighted slightly
                       towards days that produced stronger material (more/
                       better candidates), so a single quiet travel day
                       doesn't claim equal real-estate to a packed one.
  2. Place floor     — within each day, guarantee every distinct place visited
                       gets its best shot in before anything gets a second.
  3. Global top-up   — fill any slots quotas couldn't (e.g. a day with fewer
                       good candidates than its quota) from the overall best
                       leftovers, with a soft per-day ceiling so one day still
                       can't swallow the whole album.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trippy.types import ScoredPhoto

_UNKNOWN_DAY = "unknown-day"


@dataclass(frozen=True)
class DiversityConfig:
    target_count: int = 100
    # A day may absorb extra top-up slots up to (base_quota * this), so a
    # single exceptional day can shine a bit more without dominating.
    ceiling_multiplier: float = 1.6


def _day_key(photo: ScoredPhoto) -> str:
    if photo.meta.timestamp is None:
        return _UNKNOWN_DAY
    return photo.meta.timestamp.date().isoformat()


def _allocate_day_quotas(days: list[str], by_day: dict[str, list[ScoredPhoto]], target: int) -> dict[str, int]:
    if not days:
        return {}
    base = target // len(days)
    remainder = target % len(days)

    # Extra slots go to the days whose best candidate scored highest —
    # a proxy for "this was clearly an eventful, high-quality day".
    ranked_days = sorted(days, key=lambda d: by_day[d][0].score.composite, reverse=True)
    bonus_days = set(ranked_days[:remainder])

    return {day: base + (1 if day in bonus_days else 0) for day in days}


def select_diverse_set(
    candidates: list[ScoredPhoto],
    places: dict[str, str],
    config: DiversityConfig | None = None,
) -> list[ScoredPhoto]:
    """Selects up to `config.target_count` photos spanning days and places."""
    config = config or DiversityConfig()
    if not candidates:
        return []

    by_day: dict[str, list[ScoredPhoto]] = {}
    for photo in candidates:
        by_day.setdefault(_day_key(photo), []).append(photo)
    for day_photos in by_day.values():
        day_photos.sort(key=lambda p: p.score.composite, reverse=True)

    days = list(by_day.keys())
    target = min(config.target_count, len(candidates))
    day_quota = _allocate_day_quotas(days, by_day, target)

    selected_ids: set[str] = set()
    selected: list[ScoredPhoto] = []

    def take(photo: ScoredPhoto) -> None:
        selected_ids.add(photo.meta.id)
        selected.append(photo)

    # Pass 1: within each day, cover every distinct place at least once.
    day_count: dict[str, int] = {day: 0 for day in days}
    for day in days:
        quota = day_quota[day]
        seen_places: set[str] = set()
        best_per_place: dict[str, ScoredPhoto] = {}
        for photo in by_day[day]:
            place = places[photo.meta.id]
            if place not in best_per_place:
                best_per_place[place] = photo

        for place, photo in sorted(best_per_place.items(), key=lambda kv: kv[1].score.composite, reverse=True):
            if day_count[day] >= quota:
                break
            take(photo)
            day_count[day] += 1
            seen_places.add(place)

    # Pass 2: fill the rest of each day's quota with the next-best leftovers.
    for day in days:
        quota = day_quota[day]
        for photo in by_day[day]:
            if day_count[day] >= quota:
                break
            if photo.meta.id in selected_ids:
                continue
            take(photo)
            day_count[day] += 1

    # Pass 3: global top-up from whatever's left, honoring a soft per-day ceiling.
    if len(selected) < target:
        leftovers = sorted(
            (p for p in candidates if p.meta.id not in selected_ids),
            key=lambda p: p.score.composite,
            reverse=True,
        )
        ceilings = {day: max(day_quota[day], int(day_quota[day] * config.ceiling_multiplier) + 1) for day in days}
        for photo in leftovers:
            if len(selected) >= target:
                break
            day = _day_key(photo)
            if day_count[day] >= ceilings[day]:
                continue
            take(photo)
            day_count[day] += 1

        # If a global ceiling left us short (small/odd trips), drop ceilings entirely.
        if len(selected) < target:
            for photo in leftovers:
                if len(selected) >= target:
                    break
                if photo.meta.id in selected_ids:
                    continue
                take(photo)

    selected.sort(
        key=lambda p: (
            -p.score.composite,
            p.meta.timestamp is None,
            p.meta.timestamp or p.meta.id,
            p.meta.id,
        )
    )
    return selected
