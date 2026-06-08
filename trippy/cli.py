"""`trippy curate <folder> --out <dir> -n 100` — the engine's front door.

Runs the full local pipeline over a folder of trip photos and writes:
  - copies (or symlinks) of the selected photos into --out
  - a report.json with every photo's score, component breakdown and
    human-readable reasons, plus moment/cluster membership — so picks are
    inspectable and the pipeline is tunable, not a black box.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import click

from trippy.pipeline import PipelineConfig, curate_folder
from trippy.types import CurationResult, ScoredPhoto

# Reverse-geocoding (place names) is the one optional, network-dependent
# feature in Trippy -- gated entirely behind this key. Unset it and
# everything stays local/offline, exactly as before.
_MAPS_KEY_ENV_VAR = "TRIPPY_GOOGLE_MAPS_API_KEY"


def _photo_to_dict(scored: ScoredPhoto, result: CurationResult) -> dict:
    meta = scored.meta
    score = scored.score
    place_id = result.places.get(meta.id)
    return {
        "id": meta.id,
        "path": meta.path,
        "timestamp": meta.timestamp.isoformat() if meta.timestamp else None,
        "gps": {"lat": meta.gps.lat, "lon": meta.gps.lon} if meta.gps else None,
        "width": meta.width,
        "height": meta.height,
        "place": {"id": place_id, "name": result.place_names.get(place_id)} if place_id else None,
        "score": {
            "composite": round(score.composite, 4),
            "people": round(score.people, 4),
            "scenic": round(score.scenic, 4),
            "appeal": round(score.appeal, 4),
            "printability": round(score.printability, 4),
        },
        "reasons": list(score.reasons),
    }


def _build_report(result: CurationResult) -> dict:
    scored_by_id = {s.meta.id: s for s in result.all_scored}

    place_photo_ids: dict[str, list[str]] = {}
    for photo_id, place_id in result.places.items():
        place_photo_ids.setdefault(place_id, []).append(photo_id)

    return {
        "summary": {
            "total_photos": len(result.all_scored),
            "moments": len(result.moments),
            "places": len(place_photo_ids),
            "selected": len(result.selected),
        },
        "selected": [_photo_to_dict(s, result) for s in result.selected],
        "moments": [
            {
                "id": moment.id,
                "photo_count": len(moment.photo_ids),
                "photo_ids": list(moment.photo_ids),
                "representatives": list(result.representatives.get(moment.id, ())),
            }
            for moment in result.moments
        ],
        "places": [
            {
                "id": place_id,
                "name": result.place_names.get(place_id),
                "photo_count": len(photo_ids),
            }
            for place_id, photo_ids in sorted(place_photo_ids.items(), key=lambda kv: -len(kv[1]))
        ],
        "all_scores": [
            {"id": pid, "composite": round(scored_by_id[pid].score.composite, 4)}
            for pid in scored_by_id
        ],
    }


@click.group()
def main() -> None:
    """Trippy: local, deterministic best-moments photo curation."""


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="./trippy-selection",
              show_default=True, help="Directory to write selected photos + report.json into.")
@click.option("-n", "--count", "target_count", type=int, default=100, show_default=True,
              help="Target number of photos in the final selection.")
@click.option("--link/--copy", "use_symlink", default=False,
              help="Symlink instead of copying selected photos (faster, saves disk space).")
@click.option("--report-only", is_flag=True, default=False,
              help="Write report.json only; don't copy/link selected photos.")
@click.option("--maps-api-key", "maps_api_key", envvar=_MAPS_KEY_ENV_VAR, default=None,
              help=f"Google Maps API key for reverse-geocoding place names "
                   f"(optional, network-dependent; also read from ${_MAPS_KEY_ENV_VAR}). "
                   f"Omit to keep curation fully local/offline.")
def curate(folder: str, out_dir: str, target_count: int, use_symlink: bool, report_only: bool, maps_api_key: str | None) -> None:
    """Curate the best ~N moments from FOLDER into --out."""
    config = PipelineConfig(target_count=target_count, google_maps_api_key=maps_api_key)

    click.echo(f"Scanning {folder} ...")
    result = curate_folder(folder, config)

    if not result.all_scored:
        click.echo("No supported images found.")
        return

    click.echo(
        f"{len(result.all_scored)} photos -> {len(result.moments)} moments "
        f"-> {len(result.selected)} selected"
    )
    if maps_api_key:
        click.echo(f"Reverse-geocoded {len(result.place_names)} of {len(set(result.places.values()))} place(s).")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not report_only:
        for rank, scored in enumerate(result.selected, start=1):
            src = Path(scored.meta.path)
            dest = out_path / f"{rank:03d}_{src.name}"
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            if use_symlink:
                dest.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dest)

    report = _build_report(result)
    report_path = out_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    click.echo(f"Wrote {len(result.selected)} photos and report to {out_path}")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind the web server to.")
@click.option("--port", default=5050, show_default=True, type=int, help="Port to serve on.")
@click.option("--debug", is_flag=True, default=False, help="Run Flask in debug/auto-reload mode.")
@click.option("--maps-api-key", "maps_api_key", envvar=_MAPS_KEY_ENV_VAR, default=None,
              help=f"Google Maps API key for reverse-geocoding place names "
                   f"(optional, network-dependent; also read from ${_MAPS_KEY_ENV_VAR}). "
                   f"Omit to keep curation fully local/offline.")
def serve(host: str, port: int, debug: bool, maps_api_key: str | None) -> None:
    """Launch the local web UI: upload photos in a browser, see the ranked picks."""
    from trippy.web.app import main as run_server

    click.echo(f"Serving Trippy at http://{host}:{port} (Ctrl+C to stop)")
    if maps_api_key:
        click.echo("Place names: reverse-geocoding via Google Maps is enabled.")
    run_server(host=host, port=port, debug=debug, maps_api_key=maps_api_key)


if __name__ == "__main__":
    main()
