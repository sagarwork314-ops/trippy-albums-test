from collections import Counter
from datetime import datetime, timedelta

from trippy.selection.diversity import DiversityConfig, select_diverse_set
from trippy.selection.representatives import RepresentativeConfig, select_representatives
from trippy.types import Moment, PhotoFeatures, ScoreBreakdown, ScoredPhoto

from .fixtures import make_meta


def _scored(photo_id, *, day=0, hour=10, composite=0.5, phash=0):
    meta = make_meta(photo_id, timestamp=datetime(2026, 6, 1 + day, hour, 0, 0))
    features = PhotoFeatures(photo_id, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 1.0, (), (), phash)
    score = ScoreBreakdown(people=0.5, scenic=0.5, appeal=0.5, printability=1.0, composite=composite, reasons=())
    return ScoredPhoto(meta=meta, features=features, score=score)


# --- Representative selection: collapsing bursts down to their best frame(s) ---

def test_small_cluster_keeps_only_the_top_scorer():
    photos = {p.meta.id: p for p in [_scored("a", composite=0.9), _scored("b", composite=0.7), _scored("c", composite=0.6)]}
    moment = Moment("m0", ("a", "b", "c"))
    reps = select_representatives([moment], photos)
    assert reps["m0"] == ("a",)


def test_large_cluster_can_keep_a_second_distinct_strong_frame():
    photos = {}
    ids = []
    for i in range(10):
        # two near-tied top scorers with very different phashes; the rest trail far behind
        composite = 0.9 if i < 2 else 0.3
        phash = (i * 0xFFFFFFFF) if i < 2 else 0  # force large Hamming distance between the top two
        pid = f"p{i}"
        ids.append(pid)
        photos[pid] = _scored(pid, composite=composite, phash=phash)

    moment = Moment("m0", tuple(ids))
    reps = select_representatives([moment], photos, RepresentativeConfig(min_cluster_size_for_second=8))
    assert len(reps["m0"]) == 2
    assert reps["m0"][0] == "p0"  # best score wins the top slot


def test_near_identical_runner_up_is_not_kept_as_second_pick():
    photos = {}
    ids = []
    for i in range(10):
        composite = 0.9 if i < 2 else 0.3
        pid = f"p{i}"
        ids.append(pid)
        photos[pid] = _scored(pid, composite=composite, phash=0)  # identical phash for everyone

    moment = Moment("m0", tuple(ids))
    reps = select_representatives([moment], photos, RepresentativeConfig(min_cluster_size_for_second=8))
    assert len(reps["m0"]) == 1


# --- Diversity selection: spanning the whole trip, not just the busiest day ---

def test_diverse_selection_spans_every_day_present():
    candidates, places = [], {}
    for day in range(4):
        for i in range(8):
            pid = f"d{day}-{i}"
            candidates.append(_scored(pid, day=day, composite=0.9 - i * 0.05))
            places[pid] = f"place-{day}-{i % 2}"

    selected = select_diverse_set(candidates, places, DiversityConfig(target_count=12))
    days_present = {p.meta.timestamp.date() for p in selected}
    assert len(days_present) == 4
    assert len(selected) == 12


def test_diverse_selection_does_not_let_one_day_dominate():
    candidates, places = [], {}
    # Day 0: 40 great photos, all the same place (an over-photographed afternoon)
    for i in range(40):
        pid = f"big-{i}"
        candidates.append(_scored(pid, day=0, composite=0.95 - i * 0.001))
        places[pid] = "place-overrepresented"
    # Days 1-3: a handful of decent photos each, at distinct places
    for day in range(1, 4):
        for i in range(3):
            pid = f"small-{day}-{i}"
            candidates.append(_scored(pid, day=day, composite=0.6))
            places[pid] = f"place-{day}-{i}"

    selected = select_diverse_set(candidates, places, DiversityConfig(target_count=16))
    counts = Counter(p.meta.timestamp.date() for p in selected)

    busiest_day = datetime(2026, 6, 1).date()
    assert counts[busiest_day] < len(selected) / 2  # the loud day doesn't swallow the album
    assert all(count >= 1 for count in counts.values())  # every quiet day still gets in


def test_diverse_selection_covers_distinct_places_within_a_day():
    candidates, places = [], {}
    for i, place in enumerate(["A", "A", "A", "B", "C"]):
        pid = f"p{i}"
        candidates.append(_scored(pid, day=0, hour=9 + i, composite=0.9 - i * 0.05))
        places[pid] = f"place-{place}"

    selected = select_diverse_set(candidates, places, DiversityConfig(target_count=3))
    selected_places = {places[p.meta.id] for p in selected}
    assert selected_places == {"place-A", "place-B", "place-C"}


def test_diverse_selection_respects_target_count_when_fewer_candidates_exist():
    candidates = [_scored(f"p{i}", day=0, composite=0.5) for i in range(5)]
    places = {p.meta.id: "place-x" for p in candidates}
    selected = select_diverse_set(candidates, places, DiversityConfig(target_count=100))
    assert len(selected) == 5


def test_diverse_selection_returns_selected_photos_ranked_by_score():
    candidates = [
        _scored("early-low", day=0, hour=9, composite=0.2),
        _scored("late-high", day=0, hour=17, composite=0.95),
        _scored("mid-mid", day=0, hour=12, composite=0.6),
    ]
    places = {p.meta.id: f"place-{p.meta.id}" for p in candidates}

    selected = select_diverse_set(candidates, places, DiversityConfig(target_count=3))

    assert [p.meta.id for p in selected] == ["late-high", "mid-mid", "early-low"]
