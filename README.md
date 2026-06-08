# Trippy

A local, deterministic engine that turns "1000 vacation photos" into "the
~100 that actually tell the story of the trip" — without calling any LLM or
vision API. Everything runs on-device: classical computer vision (numpy/PIL)
plus a small bundled ONNX face-detection model via `onnxruntime`.

## Philosophy

A human curator doesn't score photos in isolation — they think in *moments*.
Trippy mirrors that with a two-stage approach:

1. **Is this a good photo of something worth remembering?**
   A composite "moment quality" score blends people presence & framing
   (local face detection), scenic significance (derived from GPS
   convergence — no geocoding API), and visual appeal (sharpness, exposure,
   color, composition), gated by print-worthiness (resolution, focus).

2. **Of all the good photos of the same moment, which represents it best —
   and how do we make sure the whole trip is represented?**
   Photos are clustered into "moments" (bursts/near-duplicates collapse into
   one), each moment is reduced to its best 1-2 frames, and the final set is
   chosen with day/place quotas so the result spans the whole trip rather
   than over-representing its busiest afternoon.

The whole pipeline is pure and deterministic: same photos in, same ranked
selection + explanations out. Every stage hands the next a plain,
inspectable data structure (`trippy/types.py`), so results are tunable and
explainable rather than a black box.

## Installation

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Two small ONNX models are vendored, both run fully on-device via `onnxruntime`:

- **`trippy/models/face_detector.onnx`** (~1.2MB) — UltraFace-style face
  detection -> count, size, centeredness, framing. If missing, the pipeline
  still runs; the "people" component simply falls back to the photo's own
  appeal score (see `trippy/scoring/quality.py`). Override with
  `--face-model-path` / `PipelineConfig(face_model_path=...)`.
- **`trippy/models/eye_state.onnx`** (~115KB, [OCEC](https://github.com/PINTO0309/OCEC),
  MIT) — binary "eyes open vs. closed" classifier on small eye crops, used to
  catch the single most common reason a curator rejects a people shot: someone
  blinked. Eye crop regions are *derived heuristically* from the detected face
  box (frontal-face proportions — see `trippy/scoring/eyes.py`), since there's
  no landmark model to localize eyes precisely; this degrades gracefully for
  profile shots. If missing, this signal simply stays neutral. Override with
  `--eye-model-path` / `PipelineConfig(eye_model_path=...)`.

## Usage

### Web UI

```bash
trippy serve
```

Open http://127.0.0.1:5050, upload your trip photos, pick how many should
make the final cut, and browse the ranked, explained selection in your
browser — each photo shows its composite score, people/scenic/appeal
breakdown, and the reasons it was picked. Processing is fully local and
synchronous (everything runs in the request); nothing leaves the machine.
`--host` / `--port` / `--debug` are available if you need them.

### CLI

```bash
trippy curate ./my-trip-photos --out ./best-100 -n 100
```

Writes the selected photos (copied, ranked `001_...`, `002_...`, ...) plus a
`report.json` with every photo's composite score, component breakdown
(people / scenic / appeal / printability), human-readable reasons, and
moment-cluster membership.

Useful flags:
- `-n / --count` — target size of the final selection (default 100)
- `--link` — symlink instead of copying (faster, saves disk space)
- `--report-only` — skip writing files; just produce `report.json`

## Architecture

```
trippy/
  ingest/        folder scanning, EXIF/GPS extraction (the only filesystem-touching layer)
  clustering/    perceptual hashing (dHash), GPS distance, union-find moment clustering
  scoring/       classical CV metrics, local face detection, scenic/GPS-density scoring,
                 composite "moment quality" scoring + explanations
  selection/     per-cluster representative selection, day/place diversity selection
  pipeline.py    orchestrates ingest -> cluster -> score -> select
  cli.py         `trippy curate ...` / `trippy serve`
  web/           minimal Flask UI: upload photos, browse the ranked, explained selection
```

### Moment clustering (`trippy/clustering/moment_clustering.py`)

Two photos join the same moment if they're close in time *and* space, or if
they look visually near-identical (dHash Hamming distance below a
threshold) — connected transitively via union-find, so a 30-shot burst
collapses into one cluster even if the first and last frames individually
look quite different.

### Moment quality scoring (`trippy/scoring/quality.py`)

```
composite = printability_gate * (
    w_people * people_component +   # falls back to appeal_score when no faces —
    w_scenic * scenic_score      +  # landscapes aren't structurally penalized
    w_appeal * appeal_score
)
```

- **people**: local ONNX face detection -> count, size, centeredness, framing,
  scaled by an *expression factor* (`trippy/scoring/eyes.py`) that pulls the
  score down — but only down to 60%, never to zero — when the local eye-state
  model reads someone's eyes as closed (a likely blink frame)
- **scenic**: GPS points clustered into "places"; places visited by more
  *distinct moments* score higher — a deterministic proxy for "this mattered
  on the trip". Place clusters can optionally be labelled with real names via
  Google Maps reverse-geocoding (see "Place names" below) — the only
  network-dependent, opt-in piece of the pipeline
- **appeal**: sharpness (Laplacian variance), exposure/clipping, contrast,
  colorfulness (Hasler-Süsstrunk), color-cast / white-balance (white-patch
  illuminant estimate — see below), composition (edge-density grid blending
  rule-of-thirds alignment with energy concentration), and subject isolation
  (spectral-residual saliency — see below)
- **printability_gate**: a soft multiplicative floor from resolution +
  sharpness — a technically ruined photo can't be rescued by a great moment

Color cast (`trippy/scoring/color.py::color_cast_score`) catches a defect
that colorfulness can't: an unwanted overall tint from bad white balance
(warm indoor incandescent light, cool open shade, a sensor miss) that makes
a photo look "off" even when it's technically sharp and well-exposed. It
estimates the light source's color from the *brightest non-clipped pixels*
— the "white patch" / Retinex assumption that highlights and near-white
surfaces (sky, clouds, paper, eyes) take on the illuminant's tint almost
regardless of the underlying object's true color — and checks whether that
estimate drifts from neutral gray. Two deliberate safeguards keep this from
misfiring: it samples by *brightness* rather than by "low saturation"
(a strong cast itself raises the apparent saturation of what should be
neutral content, which would blind a saturation-based search to the very
thing it's measuring), and it only trusts the sample when those brightest
spots themselves look like they were *trying* to be near-white — so a vivid
sunset or a neon-lit street (saturated even at their brightest points) reads
as a deliberate creative choice, not a white-balance defect.

Composition (`trippy/scoring/composition.py`) blends two signals from the
same 3x3 edge-density grid: **rule-of-thirds alignment** (how much visual
interest sits near the four "power points") and **energy concentration**
(how unevenly that interest is distributed across all nine cells, via
entropy). Alignment alone is easy to fool — a chaotic, cluttered background
can dump plenty of raw edge energy into the corner cells just by sheer
volume. Concentration catches that: energy spread evenly across the whole
frame reads as noise, while energy gathered in one part of it reads as
"something stands out here" — regardless of whether that part happens to
sit on a power point.

Subject isolation (`trippy/scoring/saliency.py`) goes a step further with a
genuinely different, classical signal: **spectral-residual saliency**
(Hou & Zhang, 2007) — an FFT-based technique that highlights the regions of
a frame that are statistically *surprising* relative to the rest (the way a
human eye is drawn to a face or an isolated object against a plain
background, and away from repetitive textures). No model, no training data,
fully deterministic. From the resulting saliency map we read two things: (a)
**focal concentration** — does interest gather in one place, or spread thin
with no standout subject — and (b) **focus alignment** — does that point of
interest coincide with the *sharpest* part of the frame, or is the
background crisp while the subject is soft (or absent)? That second check is
something a single global sharpness number, or the edge-density grid above,
can't catch on its own — a technically sharp photo of a cluttered scene with
a blurry subject now reads as a near-miss rather than a keeper.

Sharpness and colorfulness also get a **trip-relative percentile-rank blend**
(`trippy/scoring/normalization.py`), on top of their absolute reference-scale
scores. Absolute reference constants (e.g. "1500 Laplacian variance = tack
sharp") are calibrated against *typical* photos — but a trip shot entirely on
an overcast day, indoors, or through a phone's heavy HDR/denoise pipeline has
a different baseline; every frame can land in the same narrow band of the
absolute scale, leaving its strongest shots indistinguishable from its
weakest. Percentile rank fixes that by also asking "how does this compare to
the *other* photos from this same trip?" — closer to how a human curator
actually judges ("this is the sharpest one we've got from that overcast
morning, so it's the keeper"). It's blended with, never substituted for, the
absolute score (`percentile_blend_weight`, default 0.35): a uniformly-soft
trip's best frame still won't be called "tack sharp", but it *will* now
out-rank its siblings by more than the absolute scale alone would allow.

### Place names (optional, network-dependent)

Everything above runs fully on-device. The *one* exception, by explicit
choice, is place naming: pass a Google Maps API key and place clusters get
labelled with real names ("Shibuya, Tokyo, Japan") via the Geocoding API,
shown in the web UI and `report.json`.

```bash
trippy curate ./my-trip-photos --maps-api-key YOUR_KEY
trippy serve --maps-api-key YOUR_KEY
# or: export TRIPPY_GOOGLE_MAPS_API_KEY=YOUR_KEY
```

This calls the API **once per distinct place cluster** (its centroid), never
per photo — a 1000-photo trip touching 30 places makes ~30 calls, not 1000.
Omit the key (the default) and curation stays exactly as local/offline as
everything else; places are simply grouped, not named.

### Selection (`trippy/selection/`)

1. **Representatives**: each moment-cluster reduces to its top-scoring frame,
   with a second kept only for large clusters that genuinely contain two
   distinct, comparably-strong frames (not two near-identical ones).
2. **Diversity**: the final ~N is chosen in three passes — allocate quotas
   across days (weighted toward stronger days), guarantee every distinct
   place visited each day gets its best shot in, then top up globally by
   score with a soft per-day ceiling so no single day dominates.

## Testing

```bash
./.venv/bin/python -m pytest tests/ -q
```

Every test builds its inputs programmatically (`tests/fixtures.py`) — no
real photos, no network calls, fully deterministic.
