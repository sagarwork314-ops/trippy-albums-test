from datetime import datetime, timedelta

from trippy.pipeline import PipelineConfig, curate_folder
from trippy.types import GpsPoint

from .fixtures import draw_face, make_rgb, write_jpeg_with_exif

MUSEUM = GpsPoint(48.8606, 2.3376)
PARK = GpsPoint(48.8462, 2.3371)
LANDMARK = GpsPoint(48.8584, 2.2945)


def _build_synthetic_trip(tmp_path):
    """A small multi-day trip: a big burst (near-duplicates), a revisited
    landmark (high scenic notability), some people shots, and a couple of
    deliberately ruined (blurry / tiny) throwaways. Mirrors, in miniature,
    the "1000 photos -> 100 keepers" scenario the engine targets.
    """
    day1 = datetime(2026, 7, 1, 9, 0, 0)
    day2 = day1 + timedelta(days=1)
    paths = []

    # Realistic phone-camera resolution (~2MP) — large enough to clear the
    # print-worthiness floor, so the gate differentiates good vs. ruined shots
    # instead of flattening everything to zero.
    h, w = 1200, 1600

    # Day 1: a 9-shot burst at the museum (near-identical -> should collapse)
    for i in range(9):
        rgb = make_rgb(h, w, seed=42, sharp=True)  # identical seed -> near-duplicate frames
        ts = day1 + timedelta(seconds=i * 3)
        paths.append(write_jpeg_with_exif(tmp_path / f"d1_burst_{i:02d}.jpg", rgb, timestamp=ts, gps=MUSEUM))

    # Day 1: a couple of distinct, good single shots with people, at the park
    for i in range(3):
        rgb = draw_face(make_rgb(h, w, seed=100 + i, sharp=True, colorful=True), cx=0.4 + i * 0.05)
        ts = day1 + timedelta(hours=3, minutes=i * 20)
        paths.append(write_jpeg_with_exif(tmp_path / f"d1_park_{i}.jpg", rgb, timestamp=ts, gps=PARK))

    # Day 2: the landmark, revisited three times across the day (high scenic notability)
    for visit in range(3):
        for i in range(2):
            rgb = make_rgb(h, w, seed=200 + visit * 10 + i, sharp=True, colorful=True)
            ts = day2 + timedelta(hours=visit * 3, minutes=i * 5)
            paths.append(write_jpeg_with_exif(tmp_path / f"d2_landmark_{visit}_{i}.jpg", rgb, timestamp=ts, gps=LANDMARK))

    # Day 2: a couple of throwaways — blurry and tiny — that shouldn't survive curation
    blurry = make_rgb(h, w, seed=300, sharp=False)
    ts = day2 + timedelta(hours=1, minutes=30)
    paths.append(write_jpeg_with_exif(tmp_path / "d2_blurry.jpg", blurry, timestamp=ts, gps=LANDMARK))

    tiny = make_rgb(80, 120, seed=301, sharp=True)
    ts = day2 + timedelta(hours=1, minutes=45)
    paths.append(write_jpeg_with_exif(tmp_path / "d2_tiny.jpg", tiny, timestamp=ts, gps=LANDMARK))

    return paths


def test_pipeline_collapses_burst_and_spans_the_trip(tmp_path):
    _build_synthetic_trip(tmp_path)

    result = curate_folder(tmp_path, PipelineConfig(target_count=8))

    assert len(result.all_scored) == 20  # 9 burst + 3 park + 6 landmark + 1 blurry + 1 tiny
    # The 9-shot burst should collapse into (at most a couple of) moments —
    # certainly not 9 separate ones. 20 photos with the burst fully collapsed
    # to 1 moment yields 12 moments; anything higher means it didn't collapse.
    assert len(result.moments) <= 12

    # Selection should include photos from both days, not just the
    # heavily-bursted day-1 museum visit.
    selected_days = {p.meta.timestamp.date() for p in result.selected}
    assert len(selected_days) == 2

    # The deliberately ruined throwaways should not have made the cut.
    selected_names = {p.meta.path.split("/")[-1] for p in result.selected}
    assert "d2_blurry.jpg" not in selected_names
    assert "d2_tiny.jpg" not in selected_names


def test_pipeline_deduplicates_the_burst_to_one_or_two_representatives(tmp_path):
    _build_synthetic_trip(tmp_path)
    result = curate_folder(tmp_path, PipelineConfig(target_count=20))

    burst_moment = next(m for m in result.moments if len(m.photo_ids) >= 5)
    assert len(burst_moment.photo_ids) == 9
    assert len(result.representatives[burst_moment.id]) <= 2

    burst_in_final = [
        p for p in result.selected if "d1_burst" in p.meta.path
    ]
    assert len(burst_in_final) <= 2


def test_pipeline_is_deterministic(tmp_path):
    _build_synthetic_trip(tmp_path)
    config = PipelineConfig(target_count=8)

    first = curate_folder(tmp_path, config)
    second = curate_folder(tmp_path, config)

    assert [p.meta.id for p in first.selected] == [p.meta.id for p in second.selected]
    assert [round(p.score.composite, 6) for p in first.all_scored] == \
           [round(p.score.composite, 6) for p in second.all_scored]


def test_pipeline_explanations_are_present_for_selected_photos(tmp_path):
    _build_synthetic_trip(tmp_path)
    result = curate_folder(tmp_path, PipelineConfig(target_count=8))

    for scored in result.selected:
        assert scored.score.reasons
        assert all(isinstance(r, str) and r for r in scored.score.reasons)


def test_pipeline_handles_empty_folder(tmp_path):
    result = curate_folder(tmp_path, PipelineConfig(target_count=8))
    assert result.selected == ()
    assert result.moments == ()
    assert result.all_scored == ()
