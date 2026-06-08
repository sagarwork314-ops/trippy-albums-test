"""A small Flask front-end: upload a batch of trip photos, run the local
curation pipeline, and browse the ranked, explained selection in a browser.

Deliberately minimal — synchronous processing, in-memory result store, no
auth, no database. It's a way to *see* the engine work, not a production
photo service. Each upload gets its own folder under a temp directory so
runs don't collide and the original files remain servable for display.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for

from trippy.pipeline import PipelineConfig, curate_folder
from trippy.types import CurationResult

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "trippy-web-uploads"

# In-memory run store: run_id -> CurationResult. Fine for a single-process
# local demo; restarting the server clears it (uploaded files remain on disk
# under UPLOAD_ROOT until the OS cleans the temp dir).
_RUNS: dict[str, CurationResult] = {}

# A "1000 photos from a trip" batch can easily be several GB at full phone-camera
# resolution. This server only binds to localhost by default, so there's no
# remote-abuse concern — leave the upload size effectively uncapped.
_MAX_UPLOAD_BYTES = None


def create_app(maps_api_key: str | None = None) -> Flask:
    app = Flask(__name__)
    # Optional, network-dependent: enables real place names on the results
    # page via Google Maps reverse-geocoding. Unset (default) keeps the whole
    # run local/offline -- places are simply shown by cluster, unnamed.
    app.config["TRIPPY_MAPS_API_KEY"] = maps_api_key
    app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_BYTES
    # Each uploaded photo is its own multipart form part — Flask's default cap
    # of 1000 would reject exactly the "1000 photos from a trip" scenario this
    # engine targets, so raise it generously.
    app.config["MAX_FORM_PARTS"] = 10_000
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/curate")
    def curate():
        files = [f for f in request.files.getlist("photos") if f.filename]
        if not files:
            return redirect(url_for("index"))

        try:
            target_count = max(1, min(1000, int(request.form.get("count", 100))))
        except ValueError:
            target_count = 100

        run_id = uuid.uuid4().hex[:12]
        run_dir = UPLOAD_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        for uploaded in files:
            # Strip any client-supplied path components — only the filename matters.
            dest = run_dir / Path(uploaded.filename).name
            uploaded.save(dest)

        config = PipelineConfig(target_count=target_count, google_maps_api_key=app.config["TRIPPY_MAPS_API_KEY"])
        result = curate_folder(run_dir, config)
        _RUNS[run_id] = result

        return redirect(url_for("results", run_id=run_id))

    @app.get("/results/<run_id>")
    def results(run_id: str):
        result = _RUNS.get(run_id)
        if result is None:
            abort(404)

        # `result.selected` is ordered chronologically (so the trip "reads" in
        # order); the results page is a *ranking*, so re-sort by composite
        # score, best first, before numbering.
        by_score = sorted(result.selected, key=lambda s: s.score.composite, reverse=True)

        def place_label(photo_id: str) -> str | None:
            place_id = result.places.get(photo_id)
            if place_id is None:
                return None
            return result.place_names.get(place_id, place_id)

        ranked = [
            {
                "rank": rank,
                "filename": Path(scored.meta.path).name,
                "url": url_for("photo", run_id=run_id, filename=Path(scored.meta.path).name),
                "composite": round(scored.score.composite, 3),
                "people": round(scored.score.people, 3),
                "scenic": round(scored.score.scenic, 3),
                "appeal": round(scored.score.appeal, 3),
                "reasons": list(scored.score.reasons),
                "timestamp": scored.meta.timestamp.strftime("%Y-%m-%d %H:%M") if scored.meta.timestamp else None,
                "place": place_label(scored.meta.id),
            }
            for rank, scored in enumerate(by_score, start=1)
        ]

        # Surfaces the underlying structure: photos are timestamp-ordered,
        # clustered into moments, those moments cluster into places by GPS
        # proximity (optionally labelled via reverse-geocoding), and the
        # final selection is balanced across both days and places. This
        # summary makes that location-cluster layer visible on the page.
        selected_ids = {s.meta.id for s in result.selected}
        place_photo_ids: dict[str, list[str]] = {}
        for photo_id, place_id in result.places.items():
            place_photo_ids.setdefault(place_id, []).append(photo_id)

        places_summary = sorted(
            (
                {
                    "name": result.place_names.get(place_id, place_id),
                    "photo_count": len(photo_ids),
                    "selected_count": sum(1 for pid in photo_ids if pid in selected_ids),
                }
                for place_id, photo_ids in place_photo_ids.items()
            ),
            key=lambda p: -p["photo_count"],
        )

        return render_template(
            "results.html",
            run_id=run_id,
            photos=ranked,
            total_photos=len(result.all_scored),
            total_moments=len(result.moments),
            total_places=len(place_photo_ids),
            total_selected=len(result.selected),
            places=places_summary,
        )

    @app.get("/photos/<run_id>/<path:filename>")
    def photo(run_id: str, filename: str):
        run_dir = UPLOAD_ROOT / run_id
        if not run_dir.is_dir():
            abort(404)
        return send_from_directory(run_dir, filename)

    return app


def main(host: str = "127.0.0.1", port: int = 5050, debug: bool = False, maps_api_key: str | None = None) -> None:
    create_app(maps_api_key=maps_api_key).run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main(debug=True)
