from datetime import datetime, timedelta

import numpy as np

from trippy.scoring.color import color_cast_score, colorfulness_score
from trippy.scoring.composition import composition_score
from trippy.scoring.exposure import brightness, contrast, exposure_penalty
from trippy.scoring.faces import people_score
from trippy.scoring.quality import ScoringConfig, score_photo
from trippy.scoring.resolution import resolution_score
from trippy.scoring.scenic import ScenicConfig, assign_places, compute_scenic_scores
from trippy.scoring.sharpness import sharpness_score
from trippy.types import FaceBox, GpsPoint, Moment, PhotoFeatures

from .fixtures import draw_face, make_meta, make_rgb

PARIS = GpsPoint(48.8566, 2.3522)
NICE = GpsPoint(43.7102, 7.2620)


# --- Classical CV metrics: each should respond directionally as expected ---

def test_sharpness_ranks_sharp_above_blurred():
    sharp = make_rgb(seed=1, sharp=True)
    blurred = make_rgb(seed=1, sharp=False)
    assert sharpness_score(sharp) > sharpness_score(blurred)


def test_brightness_tracks_intended_exposure_level():
    dark = make_rgb(seed=2, brightness=0.15)
    bright = make_rgb(seed=2, brightness=0.85)
    assert brightness(dark) < brightness(bright)


def test_exposure_penalty_punishes_extremes_more_than_midtones():
    midtone = make_rgb(seed=3, brightness=0.5)
    blown_out = make_rgb(seed=3, brightness=0.97)
    assert exposure_penalty(blown_out) > exposure_penalty(midtone)


def test_contrast_is_lower_for_flatter_images():
    import numpy as np
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)
    varied = make_rgb(seed=4, sharp=True)
    assert contrast(flat) < contrast(varied)


def test_colorfulness_ranks_vivid_above_grayscale():
    vivid = make_rgb(seed=5, colorful=True)
    drab = make_rgb(seed=5, colorful=False)
    assert colorfulness_score(vivid) > colorfulness_score(drab)


def _neutral_scene(height=300, width=300, *, tint=(0, 0, 0), seed=11):
    """A scene with plenty of varied-brightness near-neutral content (walls,
    highlights, shadow) -- optionally pushed off-gray by a fixed per-channel
    `tint`, simulating a white-balance miss (warm indoor light, cool shade)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 255, (height, width), dtype=np.int64)
    channels = [np.clip(base + offset, 0, 255).astype(np.uint8) for offset in tint]
    return np.stack(channels, axis=-1)


def _vivid_sunset(height=300, width=300, seed=11):
    """A frame that's saturated orange/red top to bottom -- including its
    brightest spots -- with no near-neutral content anywhere. A legitimate
    creative color choice, not a white-balance defect."""
    rng = np.random.default_rng(seed)
    return np.stack([
        rng.integers(200, 256, (height, width), dtype=np.int64),
        rng.integers(100, 160, (height, width), dtype=np.int64),
        rng.integers(20, 80, (height, width), dtype=np.int64),
    ], axis=-1).astype(np.uint8)


def test_color_cast_score_is_neutral_for_a_well_balanced_scene():
    assert color_cast_score(_neutral_scene()) == 1.0


def test_color_cast_score_penalizes_a_systemic_tint_over_neutral_content():
    warm_cast = color_cast_score(_neutral_scene(tint=(25, 5, -25)))
    neutral = color_cast_score(_neutral_scene())
    assert neutral > warm_cast
    assert 0.0 <= warm_cast < 1.0


def test_color_cast_score_is_monotonic_in_cast_strength():
    mild = color_cast_score(_neutral_scene(tint=(10, 2, -10)))
    strong = color_cast_score(_neutral_scene(tint=(25, 5, -25)))
    assert mild > strong


def test_color_cast_score_does_not_penalize_intentionally_vivid_scenes():
    # A sunset's brightest pixels are themselves saturated orange -- there's
    # no near-white reference to judge a "cast" against, so this should read
    # as neutral rather than be mistaken for bad white balance.
    assert color_cast_score(_vivid_sunset()) == 1.0


def test_composition_score_in_unit_range():
    rgb = make_rgb(seed=6)
    assert 0.0 <= composition_score(rgb) <= 1.0


def _textured_patch(row, col, *, height=300, width=300, seed=0):
    """A flat midtone frame with a single noisy/textured patch in one 3x3
    grid cell — i.e. all the "visual interest" sits in exactly one place."""
    import numpy as np

    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    r0, r1 = row * height // 3, (row + 1) * height // 3
    c0, c1 = col * width // 3, (col + 1) * width // 3
    img[r0:r1, c0:c1] = rng.integers(0, 256, (r1 - r0, c1 - c0, 3), dtype=np.int32).astype(np.uint8)
    return img


def _uniformly_busy_image(*, height=300, width=300, seed=0):
    """Edge energy smeared evenly across the whole frame — a cluttered
    background with no isolated subject."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.int32).astype(np.uint8)


def test_composition_score_prefers_subject_near_a_power_point_over_dead_center():
    near_power_point = _textured_patch(row=0, col=0)
    dead_center = _textured_patch(row=1, col=1)
    assert composition_score(near_power_point) > composition_score(dead_center)


def test_composition_score_is_not_fooled_by_uniformly_busy_clutter():
    # Same kind of edge energy, but smeared across all nine cells rather than
    # isolated on a single subject near a power point -- raw edge volume in
    # the corner cells shouldn't be enough to win on its own.
    isolated_subject = _textured_patch(row=0, col=2)
    uniform_clutter = _uniformly_busy_image()
    assert composition_score(isolated_subject) > composition_score(uniform_clutter)


def test_resolution_score_rewards_higher_megapixels():
    # All three sit above the "unprintable" floor but span the acceptable->good
    # range, so the metric should differentiate rather than floor them all to 0.
    assert resolution_score(4000, 3000) > resolution_score(2000, 1500) > resolution_score(1000, 750)


def test_resolution_score_penalizes_extreme_aspect_ratios():
    standard = resolution_score(3000, 2000)
    panorama = resolution_score(6000, 800)
    assert standard > panorama


# --- Faces / people score ---

def test_people_score_zero_with_no_faces():
    assert people_score(()) == 0.0


def test_people_score_prefers_large_centered_well_framed_faces():
    centered = (FaceBox(x=0.35, y=0.3, width=0.3, height=0.4, confidence=0.9),)
    tiny_corner = (FaceBox(x=0.02, y=0.02, width=0.04, height=0.04, confidence=0.9),)
    assert people_score(centered) > people_score(tiny_corner)


def test_people_score_rewards_group_shots_over_solo_when_quality_equal():
    # Build faces so each individual face has *identical* size/confidence and
    # is equidistant from frame-center on average — symmetric placement means
    # the group's average per-face quality matches the solo's exactly, so the
    # only thing that can differentiate them is the multi-face bonus.
    def face_at(cx, cy, size=0.2, confidence=0.9):
        return FaceBox(x=cx - size / 2, y=cy - size / 2, width=size, height=size, confidence=confidence)

    solo = (face_at(0.3, 0.5),)
    group = (face_at(0.3, 0.5), face_at(0.7, 0.5))
    assert people_score(group) > people_score(solo)


# --- Scenic scoring (GPS-density based, no geocoding API) ---

def test_scenic_score_is_neutral_without_gps():
    metas = [make_meta("a", gps=None)]
    scores = compute_scenic_scores(metas, moments=[Moment("m0", ("a",))])
    assert scores["a"] == 0.5


def test_scenic_score_rewards_places_visited_by_more_distinct_moments():
    base = datetime(2026, 6, 1, 9, 0, 0)
    metas = [
        make_meta("hub-1", timestamp=base, gps=PARIS),
        make_meta("hub-2", timestamp=base + timedelta(hours=2), gps=GpsPoint(PARIS.lat + 0.0005, PARIS.lon)),
        make_meta("hub-3", timestamp=base + timedelta(hours=5), gps=GpsPoint(PARIS.lat - 0.0005, PARIS.lon)),
        make_meta("oneoff", timestamp=base + timedelta(hours=1), gps=NICE),
    ]
    moments = [Moment("m0", ("hub-1",)), Moment("m1", ("hub-2",)), Moment("m2", ("hub-3",)), Moment("m3", ("oneoff",))]

    scores = compute_scenic_scores(metas, moments, ScenicConfig(place_radius_meters=300))
    assert scores["hub-1"] == scores["hub-2"] == scores["hub-3"]
    assert scores["hub-1"] > scores["oneoff"]


def test_assign_places_groups_nearby_gps_and_isolates_missing_gps():
    metas = [
        make_meta("p1", gps=PARIS),
        make_meta("p2", gps=GpsPoint(PARIS.lat + 0.0003, PARIS.lon)),
        make_meta("p3", gps=NICE),
        make_meta("p4", gps=None),
        make_meta("p5", gps=None),
    ]
    places = assign_places(metas, ScenicConfig(place_radius_meters=300))
    assert places["p1"] == places["p2"]
    assert places["p1"] != places["p3"]
    assert places["p4"] != places["p5"]  # GPS-less photos never get falsely grouped together


# --- Composite scoring ---

def _features(**overrides) -> PhotoFeatures:
    base = dict(
        photo_id="x", sharpness=0.8, brightness=0.5, contrast=0.5, colorfulness=0.6,
        color_cast=0.8, composition=0.6, subject_isolation=0.6, exposure_penalty=0.1,
        resolution_score=0.9, faces=(), eye_openness=(), phash=0,
    )
    base.update(overrides)
    return PhotoFeatures(**base)


def test_composite_score_penalizes_blur_and_low_resolution():
    good = score_photo(_features(sharpness=0.9, resolution_score=0.95), scenic_score=0.5)
    ruined = score_photo(_features(sharpness=0.05, resolution_score=0.1), scenic_score=0.5)
    assert good.composite > ruined.composite
    assert ruined.printability < good.printability


def test_composite_score_rewards_well_framed_people():
    no_people = score_photo(_features(faces=()), scenic_score=0.5)
    great_people = (FaceBox(x=0.35, y=0.25, width=0.3, height=0.45, confidence=0.95),)
    with_people = score_photo(_features(faces=great_people), scenic_score=0.5)
    assert with_people.people > no_people.people
    # Composite should not collapse just because a photo has no people:
    assert no_people.composite > 0.0


def test_composite_score_reflects_scenic_significance():
    plain = score_photo(_features(), scenic_score=0.3)
    notable = score_photo(_features(), scenic_score=0.95)
    assert notable.composite > plain.composite
    assert notable.scenic > plain.scenic


def test_reasons_are_explainable_and_nonempty():
    breakdown = score_photo(_features(sharpness=0.05), scenic_score=0.5)
    assert breakdown.reasons
    assert any("blur" in r or "soft" in r for r in breakdown.reasons)
