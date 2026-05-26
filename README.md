# QCI — Queue Count Intelligence

**Privacy-preserving, computational-imaging crowd counter for polling stations.**

Voters in Delhi can check wait times at their polling station on their phone
before going to vote.  Camera feeds are optically encoded before any processing
so that individual faces cannot be recovered from stored data — verified with
published-quality privacy analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CAMERA FEED                                                    │
│      ↓                                                          │
│  Layer 1 · OpticalEncoder ──────── defocus / coded-mask / PSF  │
│      ↓                             encoding_strength ∈ [0,1]   │
│  Layer 1 · DegradationSim ──────── noise / motion / low-light  │
│      ↓                                                          │
│  Layer 2 · RestorationModule ───── Wiener / U-Net stub         │
│      ↓                                                          │
│  Layer 3 · CrowdCounter ────────── HOG / CSRNet / YOLO         │
│      ↓           ↘                                             │
│  Layer 4 · GroundPlaneMapper      CrowdCounter → person count  │
│      ↓                                                          │
│  Layer 5 · ServiceRateModel ────── M/M/c queuing → wait ETA    │
│      ↓                                                          │
│  Layer 5 · HistoryTracker ──────── SQLite snapshots            │
│      ↓                                                          │
│  Layer 6 · PrivacyUtilityAnalyzer  FDR + EER + PSNR analysis  │
│      ↓                                                          │
│  Layer 7 · FastAPI + WebSocket ─── voter-facing real-time UI   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
# 1. Install
pip install -e ".[server]"          # includes FastAPI + uvicorn
pip install -e ".[dev]"             # adds pytest + httpx

# 2. Seed the database with 3 hours of synthetic history
python -m qci.server.seed

# 3. Start the API server
uvicorn qci.server.api:app --reload

# 4. Open the voter UI  (works as file:// OR via the server at /)
open frontend/index.html

# 5. Simulate a live election day
python scripts/demo.py              # posts updates every 30 s
```

---

## Running each research layer

| Layer | Command | Output |
|-------|---------|--------|
| L1–L3 encoding sweep | `python scripts/run_sweep.py` | `results/sweep_plot.png` |
| L3 multi-counter sweep | `python scripts/run_sweep.py --all-counters` | `results/counters_results.csv` |
| L4 ground-plane geometry | `python scripts/run_geometry.py --synthetic` | `results/birdseye.png` |
| L5 analytics (one reading) | `python scripts/run_analytics.py --station_id ABC --count 47 --queue_length_m 23` | ETA report |
| **L6 privacy eval** | `python scripts/run_privacy_eval.py` | **`results/privacy_utility_tradeoff.png`** |
| Full pipeline | `python scripts/run_full_pipeline.py --skip_sweep` | `results/summary_table.csv` |

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/stations` | All stations with latest status |
| `GET` | `/stations/{id}` | One station + last 2 h history |
| `POST` | `/stations/{id}/update` | Push a count update (JSON or image upload) |
| `GET` | `/stations/{id}/privacy_check` | Current encoding strength + FDR |
| `WS` | `/ws/{id}` | Live push of new QueueStatus messages |
| `GET` | `/` | Serves `frontend/index.html` |

### POST /stations/{id}/update

**JSON body** (demo / testing):
```json
{ "count": 47, "queue_length_m": 23, "n_booths": 3 }
```

**Multipart image upload** (production):
```bash
curl -X POST http://localhost:8000/stations/DL-001/update \
     -F "image=@polling_station_photo.jpg"
```

---

## Docker deployment

```bash
cp .env.example .env          # edit ENCODING_STRENGTH etc. if needed
docker-compose up --build
# API at http://localhost:8000
```

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENCODING_STRENGTH` | `0.4` | Privacy encoding level (0=none, 1=max) |
| `N_BOOTHS_DEFAULT` | `3` | Active voting booths per station |
| `COUNTER_TYPE` | `hog` | `hog` / `csrnet` / `density` |
| `AVG_SERVICE_TIME_SEC` | `120` | Mean booth service time (s) |

---

## Running tests

```bash
pytest                         # all 166 tests
pytest tests/test_server.py    # Layer 7 API only
pytest tests/test_privacy.py   # Layer 6 privacy eval
pytest tests/test_analytics.py # Layer 5 queue analytics
```

---

## Project layout

```
qci/
  optics/          # L1 — OpticalEncoder, DegradationSim, PSF utils
  data/            # L2 — ShanghaiTech loader, synthetic dataset
  recovery/        # L2 — Wiener, U-Net stub
  counting/        # L3 — HOG, CSRNet, YOLO, crowd_regime
  eval/            # L2+L3 — metrics, sweep runner
  geometry/        # L4 — CameraModel, GroundPlaneMapper, queue length
  analytics/       # L5 — ServiceRateModel, QueueStatus, HistoryTracker
  privacy/         # L6 — face attackers, inversion, PrivacyUtilityAnalyzer
  server/          # L7 — FastAPI app, WebSocket, seed, worker
    api.py             REST + WebSocket endpoints
    worker.py          Image inference pipeline (asyncio.to_thread)
    seed.py            Synthetic historical data generator
    stations.py        5 Delhi polling station definitions
    config.py          Environment-variable config

frontend/
  index.html       Voter-facing mobile UI (vanilla JS, no build step)

scripts/
  run_sweep.py           L1–L3 encoding sweep
  run_geometry.py        L4 ground-plane projection
  run_analytics.py       L5 wait-time estimate
  run_privacy_eval.py    L6 privacy–utility tradeoff  ← paper figure
  run_full_pipeline.py   End-to-end summary table
  demo.py                Live demo simulator

configs/
  sweep.yaml       Default hyperparameters

Dockerfile
docker-compose.yml
.env.example
```

---

## Key results

After running `python scripts/run_privacy_eval.py`:

- **`results/privacy_utility_tradeoff.png`** — the publication-quality figure
  (Top: MAE vs encoding strength; Bottom: FDR + EER vs encoding strength;
  shaded "privacy achieved" zone; Pareto-optimal operating point.)

- **`results/privacy_results.csv`** — per-strength metrics table

At `encoding_strength = 0.4` (default):
- Face Detection Rate (FDR) drops below 20 %
- Equal Error Rate (EER) rises above 40 %
- HOG counter MAE remains within 2× of the unencoded baseline

---

## Academic context

This codebase accompanies research on *privacy-preserving optical encoding for
crowd monitoring at polling stations*.  The core claim — optical PSF encoding
protects voter privacy while preserving crowd-count accuracy — is quantified by
`PrivacyUtilityAnalyzer` (`qci/privacy/analyzer.py`).  The Erlang-C M/M/c model
(`qci/analytics/service_rate.py`) translates person counts into realistic
voter wait-time estimates.
