"""QCI FastAPI server — real-time voter-facing queue status API."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from qci.analytics import HistoryTracker, QueueStatus, ServiceRateModel
from qci.server.config import get_config
from qci.server.stations import DELHI_STATIONS, DELHI_STATIONS_BY_ID, get_station_meta

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state (single-process)
# ---------------------------------------------------------------------------

_history: Optional[HistoryTracker] = None
_privacy_cache: Dict[float, float] = {}   # strength → FDR
_ws_queues: Dict[str, List[asyncio.Queue]] = {}   # station_id → subscriber queues


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _history, _privacy_cache

    cfg = get_config()

    # Initialise SQLite tracker
    _history = HistoryTracker(db_path=cfg.db_path)
    log.info("HistoryTracker opened: %s", cfg.db_path)

    # Load privacy results CSV for /privacy_check
    priv_csv = Path(cfg.privacy_results_csv)
    if priv_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(priv_csv)
            if {"strength", "fdr"}.issubset(df.columns):
                _privacy_cache = dict(zip(df["strength"].tolist(), df["fdr"].tolist()))
                log.info("Privacy cache loaded: %d entries", len(_privacy_cache))
        except Exception as exc:
            log.warning("Could not load privacy CSV: %s", exc)

    yield

    if _history:
        _history.close()
        log.info("HistoryTracker closed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="QCI — Queue Count Intelligence",
    description="Privacy-preserving real-time crowd count for polling stations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_dict(status: QueueStatus) -> dict:
    return json.loads(status.to_json())


def _closest_fdr(strength: float) -> Optional[float]:
    if not _privacy_cache:
        return None
    closest = min(_privacy_cache.keys(), key=lambda k: abs(k - strength))
    return _privacy_cache[closest]


async def _broadcast(station_id: str, status: QueueStatus) -> None:
    if station_id not in _ws_queues:
        return
    msg = status.to_json()
    stale: List[asyncio.Queue] = []
    for q in _ws_queues[station_id]:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            stale.append(q)
    for q in stale:
        try:
            _ws_queues[station_id].remove(q)
        except ValueError:
            pass


def _assert_tracker() -> HistoryTracker:
    if _history is None:
        raise HTTPException(503, "Database not ready")
    return _history


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend() -> str:
    frontend = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if frontend.exists():
        return frontend.read_text()
    return "<h1>Frontend not built — open frontend/index.html directly.</h1>"


@app.get("/stations")
async def list_stations() -> JSONResponse:
    """Return all known stations with their latest QueueStatus."""
    tracker = _assert_tracker()
    result = []
    for s in DELHI_STATIONS:
        sid = s["id"]
        latest = tracker.get_latest(sid)
        result.append({
            **s,
            "latest": _status_dict(latest) if latest else None,
        })
    return JSONResponse({"stations": result})


@app.get("/stations/{station_id}")
async def get_station(station_id: str) -> JSONResponse:
    """Return latest QueueStatus + last 2 hours of history for one station."""
    tracker = _assert_tracker()
    latest = tracker.get_latest(station_id)
    history = tracker.get_history(station_id, hours=2.0)
    meta = get_station_meta(station_id)
    return JSONResponse({
        **meta,
        "latest": _status_dict(latest) if latest else None,
        "history": [_status_dict(s) for s in history],
    })


@app.post("/stations/{station_id}/update")
async def update_station(station_id: str, request: Request) -> JSONResponse:
    """Accept a crowd-count update.

    Supports two content types:

    * ``multipart/form-data`` with an ``image`` file field — runs the full
      ML inference pipeline.
    * ``application/json`` with ``{count, queue_length_m, n_booths}`` —
      skips inference and stores the values directly (demo / testing).
    """
    tracker = _assert_tracker()
    cfg = get_config()
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        image_field = form.get("image")
        if image_field is not None:
            # Full inference pipeline in a thread
            from qci.server.worker import process_image_bytes
            image_bytes = await image_field.read()
            status = await asyncio.to_thread(process_image_bytes, station_id, image_bytes, cfg)
        else:
            # Multipart form without an image file — read count/queue fields
            try:
                count = float(form.get("count", 0))
                queue_length_m = float(form.get("queue_length_m", count * 0.5))
                n_booths = int(form.get("n_booths", cfg.n_booths_default))
            except (TypeError, ValueError) as exc:
                raise HTTPException(422, f"Invalid form field: {exc}") from exc
            status = _build_direct_status(station_id, count, queue_length_m, n_booths, cfg)
    else:
        # JSON body
        try:
            data = await request.json()
        except Exception as exc:
            raise HTTPException(422, f"Could not parse JSON body: {exc}") from exc
        count = float(data.get("count", 0))
        queue_length_m = float(data.get("queue_length_m", count * 0.5))
        n_booths = int(data.get("n_booths", cfg.n_booths_default))
        status = _build_direct_status(station_id, count, queue_length_m, n_booths, cfg)

    tracker.insert(status)
    await _broadcast(station_id, status)
    return JSONResponse(_status_dict(status))


def _build_direct_status(
    station_id: str,
    count: float,
    queue_length_m: float,
    n_booths: int,
    cfg,
) -> QueueStatus:
    model = ServiceRateModel(
        n_booths=n_booths,
        avg_service_time_sec=cfg.avg_service_time_sec,
    )
    wait = model.estimate_wait(count)
    return QueueStatus.create(
        station_id=station_id,
        person_count=count,
        queue_length_m=queue_length_m,
        crowd_density=0.0,
        wait_estimate=wait,
        encoding_strength=cfg.encoding_strength,
    )


@app.get("/stations/{station_id}/privacy_check")
async def privacy_check(station_id: str) -> JSONResponse:
    """Return current encoding strength and known face detection rate."""
    cfg = get_config()
    strength = cfg.encoding_strength
    fdr = _closest_fdr(strength)
    return JSONResponse({
        "station_id": station_id,
        "encoding_strength": strength,
        "face_detection_rate": fdr,
        "privacy_protected": (fdr < 0.20) if fdr is not None else None,
        "note": (
            "privacy_results.csv not found — run scripts/run_privacy_eval.py to generate it."
            if fdr is None
            else f"FDR={fdr:.2%} at encoding_strength={strength:.2f}"
        ),
    })


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/{station_id}")
async def websocket_endpoint(websocket: WebSocket, station_id: str) -> None:
    """Push new QueueStatus JSON whenever the station is updated."""
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _ws_queues.setdefault(station_id, []).append(q)
    try:
        while True:
            msg = await asyncio.wait_for(q.get(), timeout=30.0)
            await websocket.send_text(msg)
    except asyncio.TimeoutError:
        # Heartbeat: send a ping to keep connection alive
        try:
            await websocket.send_text(json.dumps({"ping": True}))
        except Exception:
            pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("WebSocket %s closed: %s", station_id, exc)
    finally:
        try:
            _ws_queues[station_id].remove(q)
        except (KeyError, ValueError):
            pass
