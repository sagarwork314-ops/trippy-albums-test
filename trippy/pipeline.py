"""Orchestrates the full curation pipeline: ingest -> cluster -> score -> select.

Pure and deterministic given the same input folder + config: no network
calls, no LLM/vision-API dependency, no randomness. Every stage hands the
next a plain, inspectable data structure (see trippy.types), which is what
makes the whole thing easy to test, tune, and explain.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from trippy.clustering.moment_clustering import ClusteringConfig, cluster_moments
from trippy.ingest.loader import LoadedPhoto, load_folder
from trippy.scoring.eyes import OnnxEyeStateClassifier
from trippy.scoring.faces import OnnxFaceDetector
from trippy.scoring.geocoding import GoogleMapsGeocoder, name_places
from trippy.scoring.normalization import percentile_ranks
from trippy.scoring.quality import ScoringConfig, extract_features, score_photo
from trippy.scoring.scenic import ScenicConfig, assign_places, compute_scenic_scores
from trippy.selection.diversity import DiversityConfig, select_diverse_set
from trippy.selection.representatives import RepresentativeConfig, select_representatives
from trippy.types import CurationResult, Moment, PhotoFeatures, ScoredPhoto

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class PipelineConfig:
    target_count: int = 100
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    scenic: ScenicConfig = field(default_factory=ScenicConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    representatives: RepresentativeConfig = field(default_factory=RepresentativeConfig)
    diversity_ceiling_multiplier: float = 1.6
    face_model_path: str | None = None
    eye_model_path: str | None = None
    # Optional, network-dependent: when set, place clusters are labelled with
    # real place names via the Google Maps Geocoding API (one call per
    # distinct place, not per photo). Everything else in the pipeline stays
    # local/offline; this is the one opt-in exception. Leave unset (the
    # default) to keep Trippy fully local — places remain unnamed.
    google_maps_api_key: str | None = None


def _report_progress(progress_callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(max(0.0, min(1.0, fraction)), message)


def _extract_all_features(
    loaded: list[LoadedPhoto],
    config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, PhotoFeatures]:
    """Runs the expensive, cluster-independent measurements once per photo:
    classical CV metrics + local face detection + perceptual hashing."""
    detector = OnnxFaceDetector(config.face_model_path)
    eye_classifier = OnnxEyeStateClassifier(config.eye_model_path)
    features: dict[str, PhotoFeatures] = {}
    total = len(loaded)
    for index, lp in enumerate(loaded, start=1):
        features[lp.meta.id] = extract_features(lp.meta, lp.rgb, detector, eye_classifier)
        _report_progress(
            progress_callback,
            0.20 + 0.52 * (index / total),
            f"Scoring photo {index} of {total}",
        )
    return features


def curate_folder(
    folder: str | Path,
    config: PipelineConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CurationResult:
    """Runs the full pipeline over every supported image in `folder`."""
    config = config or PipelineConfig()

    _report_progress(progress_callback, 0.05, "Loading images")
    loaded = load_folder(folder)
    _report_progress(progress_callback, 0.15, f"Loaded {len(loaded)} image(s)")
    return curate_photos(loaded, config, progress_callback=progress_callback)


def curate_photos(
    loaded: list[LoadedPhoto],
    config: PipelineConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CurationResult:
    """Runs the pipeline over already-decoded photos (used by tests / callers
    that load images themselves, e.g. to share decoding work)."""
    config = config or PipelineConfig()
    if not loaded:
        _report_progress(progress_callback, 1.0, "Done")
        return CurationResult(selected=(), moments=(), representatives={}, all_scored=(), places={}, place_names={})

    metas = [lp.meta for lp in loaded]
    metas_by_id = {m.id: m for m in metas}

    # Stage 1: per-photo measurements (CV metrics, faces, perceptual hash).
    # These don't depend on clustering, so we compute them exactly once.
    features_by_id = _extract_all_features(loaded, config, progress_callback)

    # Stage 2: cluster into moments (uses perceptual hashes from stage 1).
    _report_progress(progress_callback, 0.76, "Clustering moments")
    moments = cluster_moments(metas, features_by_id, config.clustering)

    # Stage 3: scenic scores depend on moment membership ("notability" =
    # distinct moments converging on a place), so they can only be computed now.
    _report_progress(progress_callback, 0.82, "Scoring places")
    places = assign_places(metas, config.scenic)
    scenic_by_id = compute_scenic_scores(metas, moments, config.scenic, places=places)

    # Optional: label place clusters with real names via Google Maps (one
    # lookup per distinct place, never per photo). No-op if no key is set.
    geocoder = GoogleMapsGeocoder(config.google_maps_api_key)
    gps_by_id = {m.id: m.gps for m in metas if m.gps is not None}
    place_names = name_places(places, gps_by_id, geocoder)

    # Trip-relative percentile ranks (see scoring/normalization.py): blended
    # into sharpness/colorfulness so the standout shots of an atypical batch
    # (overcast day, indoors, heavy phone HDR) separate from their siblings
    # by more than the absolute reference scale alone would allow.
    sharpness_percentiles = percentile_ranks({pid: f.sharpness for pid, f in features_by_id.items()})
    colorfulness_percentiles = percentile_ranks({pid: f.colorfulness for pid, f in features_by_id.items()})

    # Stage 4: combine into the final composite score per photo.
    _report_progress(progress_callback, 0.90, "Combining scores")
    scored: dict[str, ScoredPhoto] = {}
    for photo_id, features in features_by_id.items():
        breakdown = score_photo(
            features, scenic_by_id[photo_id], config.scoring,
            relative_sharpness=sharpness_percentiles[photo_id],
            relative_colorfulness=colorfulness_percentiles[photo_id],
        )
        scored[photo_id] = ScoredPhoto(meta=metas_by_id[photo_id], features=features, score=breakdown)

    _report_progress(progress_callback, 0.95, "Choosing final images")
    representatives = select_representatives(moments, scored, config.representatives)

    candidate_ids: list[str] = []
    candidate_id_set: set[str] = set()
    for picks in representatives.values():
        for pid in picks:
            if pid in candidate_id_set:
                continue
            candidate_ids.append(pid)
            candidate_id_set.add(pid)

    # The representative set is intentionally compact for normal curation, but
    # if the user asks for more photos than that compact pool contains (for
    # example, all 95 uploaded images), fill with the next-best non-represented
    # photos so the requested count is honored up to the number of inputs.
    target_candidates = min(max(config.target_count, 0), len(scored))
    if len(candidate_ids) < target_candidates:
        for photo_id, scored_photo in sorted(scored.items(), key=lambda item: item[1].score.composite, reverse=True):
            if photo_id in candidate_id_set:
                continue
            candidate_ids.append(photo_id)
            candidate_id_set.add(photo_id)
            if len(candidate_ids) >= target_candidates:
                break

    candidates = [scored[pid] for pid in candidate_ids]

    diversity_config = DiversityConfig(
        target_count=config.target_count,
        ceiling_multiplier=config.diversity_ceiling_multiplier,
    )
    selected = select_diverse_set(candidates, places, diversity_config)

    _report_progress(progress_callback, 1.0, "Done")
    return CurationResult(
        selected=tuple(selected),
        moments=tuple(moments),
        representatives=representatives,
        all_scored=tuple(scored.values()),
        places=places,
        place_names=place_names,
    )
