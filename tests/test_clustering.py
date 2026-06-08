from datetime import datetime, timedelta

from trippy.clustering.geo import haversine_meters
from trippy.clustering.moment_clustering import ClusteringConfig, cluster_moments
from trippy.clustering.perceptual_hash import dhash, hamming_distance
from trippy.clustering.union_find import UnionFind
from trippy.types import GpsPoint, PhotoFeatures

from .fixtures import draw_face, make_meta, make_rgb

PARIS = GpsPoint(48.8566, 2.3522)
NICE = GpsPoint(43.7102, 7.2620)


def _features_for(photo_id: str, rgb) -> PhotoFeatures:
    return PhotoFeatures(
        photo_id=photo_id, sharpness=0.5, brightness=0.5, contrast=0.5, colorfulness=0.5,
        color_cast=0.5, composition=0.5, subject_isolation=0.5, exposure_penalty=0.1,
        resolution_score=1.0, faces=(), eye_openness=(), phash=dhash(rgb),
    )


def test_dhash_identical_images_have_zero_distance():
    rgb = make_rgb(seed=1)
    assert hamming_distance(dhash(rgb), dhash(rgb)) == 0


def test_dhash_distinguishes_different_images():
    a = make_rgb(seed=1)
    b = draw_face(make_rgb(seed=2), cx=0.5, cy=0.5, scale=0.5)
    assert hamming_distance(dhash(a), dhash(b)) > 10


def test_haversine_zero_for_same_point():
    assert haversine_meters(PARIS, PARIS) == 0.0


def test_haversine_orders_distances_sensibly():
    near = GpsPoint(PARIS.lat + 0.001, PARIS.lon + 0.001)
    assert haversine_meters(PARIS, near) < haversine_meters(PARIS, NICE)


def test_union_find_groups_transitively():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    groups = sorted((sorted(g) for g in uf.groups()))
    assert groups == [[0, 1, 2], [3, 4]]


def test_burst_collapses_into_single_moment():
    """30 shots, 3 seconds apart, same GPS -> one moment, not 30."""
    base = datetime(2026, 6, 1, 10, 0, 0)
    metas, features = [], {}
    for i in range(10):
        rgb = make_rgb(seed=100, sharp=True)  # same seed -> visually near-identical burst frames
        pid = f"burst-{i}"
        metas.append(make_meta(pid, timestamp=base + timedelta(seconds=i * 3), gps=PARIS))
        features[pid] = _features_for(pid, rgb)

    moments = cluster_moments(metas, features)
    assert len(moments) == 1
    assert len(moments[0].photo_ids) == 10


def test_temporally_and_spatially_separate_shots_form_distinct_moments():
    base = datetime(2026, 6, 1, 10, 0, 0)
    metas, features = [], {}

    # Morning in Paris
    for i in range(3):
        pid = f"paris-{i}"
        rgb = make_rgb(seed=200 + i)
        metas.append(make_meta(pid, timestamp=base + timedelta(minutes=i), gps=PARIS))
        features[pid] = _features_for(pid, rgb)

    # Afternoon in Nice, hours later, far away, visually distinct content
    for i in range(3):
        pid = f"nice-{i}"
        rgb = draw_face(make_rgb(seed=300 + i), cx=0.3 + i * 0.1)
        metas.append(make_meta(pid, timestamp=base + timedelta(hours=6, minutes=i), gps=NICE))
        features[pid] = _features_for(pid, rgb)

    moments = cluster_moments(metas, features)
    moment_ids_by_photo = {pid: m.id for m in moments for pid in m.photo_ids}

    assert moment_ids_by_photo["paris-0"] == moment_ids_by_photo["paris-1"] == moment_ids_by_photo["paris-2"]
    assert moment_ids_by_photo["nice-0"] == moment_ids_by_photo["nice-1"] == moment_ids_by_photo["nice-2"]
    assert moment_ids_by_photo["paris-0"] != moment_ids_by_photo["nice-0"]


def test_visual_similarity_clusters_even_without_gps():
    """Same dHash, no location data at all -> still recognized as one moment."""
    base = datetime(2026, 6, 1, 10, 0, 0)
    metas, features = [], {}
    for i in range(4):
        pid = f"nogps-{i}"
        rgb = make_rgb(seed=999, sharp=True)
        metas.append(make_meta(pid, timestamp=base + timedelta(seconds=i * 2), gps=None))
        features[pid] = _features_for(pid, rgb)

    config = ClusteringConfig(phash_threshold=10)
    moments = cluster_moments(metas, features, config)
    assert len(moments) == 1
    assert len(moments[0].photo_ids) == 4
