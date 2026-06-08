from datetime import datetime
from io import BytesIO
import time

from trippy.types import CurationResult, PhotoFeatures, ScoreBreakdown, ScoredPhoto
from trippy.web import app as web_app
from trippy.web.app import _RUNS, create_app

from .fixtures import make_meta


def _scored(photo_id: str, filename: str, *, composite: float, timestamp: datetime) -> ScoredPhoto:
    meta = make_meta(photo_id, path=f"/tmp/{filename}", timestamp=timestamp)
    features = PhotoFeatures(photo_id, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 1.0, (), (), 0)
    score = ScoreBreakdown(people=0.5, scenic=0.5, appeal=0.5, printability=1.0, composite=composite, reasons=())
    return ScoredPhoto(meta=meta, features=features, score=score)


def test_results_page_ranks_selected_photos_by_score_not_timestamp():
    app = create_app()
    run_id = "score-sort"
    _RUNS[run_id] = CurationResult(
        selected=(
            _scored("early-low", "early-low.jpg", composite=0.2, timestamp=datetime(2026, 6, 1, 9)),
            _scored("late-high", "late-high.jpg", composite=0.95, timestamp=datetime(2026, 6, 1, 17)),
            _scored("mid-mid", "mid-mid.jpg", composite=0.6, timestamp=datetime(2026, 6, 1, 12)),
        ),
        moments=(),
        representatives={},
        all_scored=(),
    )

    try:
        response = app.test_client().get(f"/results/{run_id}")
        html = response.get_data(as_text=True)
    finally:
        _RUNS.pop(run_id, None)

    assert response.status_code == 200
    assert html.index("late-high.jpg") < html.index("mid-mid.jpg") < html.index("early-low.jpg")
    assert "selected, ranked by score" in html


def test_index_page_has_processing_progress_bar():
    response = create_app().test_client().get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="progress-bar"' in html
    assert 'role="progressbar"' in html
    assert "pollProgress" in html


def test_json_curate_starts_background_run_and_reports_progress(monkeypatch):
    def fake_curate_folder(folder, config, progress_callback=None):
        assert config.target_count == 3
        if progress_callback is not None:
            progress_callback(0.5, "Halfway")
        return CurationResult(selected=(), moments=(), representatives={}, all_scored=())

    monkeypatch.setattr(web_app, "curate_folder", fake_curate_folder)

    client = create_app().test_client()
    response = client.post(
        "/curate",
        data={"count": "3", "photos": (BytesIO(b"fake image bytes"), "photo.jpg")},
        content_type="multipart/form-data",
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 202
    payload = response.get_json()
    run_id = payload["run_id"]

    try:
        progress = {}
        for _ in range(100):
            progress_response = client.get(payload["progress_url"])
            progress = progress_response.get_json()
            if progress["status"] == "done":
                break
            time.sleep(0.01)

        assert progress["status"] == "done"
        assert progress["percent"] == 100
        assert progress["results_url"] == f"/results/{run_id}"
    finally:
        _RUNS.pop(run_id, None)
        web_app._PROGRESS.pop(run_id, None)
