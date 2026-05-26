#!/usr/bin/env python3
"""demo.py — simulates a live election day without a real camera.

Every 30 seconds, picks a random polling station, computes a realistic
synthetic crowd count based on the time of day, POSTs to the QCI API, and
prints what the voter would see on their phone.

Run alongside the server::

    # Terminal 1
    uvicorn qci.server.api:app --reload

    # Terminal 2
    python scripts/demo.py

Optional arguments::

    --api        API base URL (default: http://localhost:8000)
    --interval   Seconds between posts (default: 30)
    --once       Post once to each station then exit (useful for seeding)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random
import sys
import time
from pathlib import Path

# Allow running from project root without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed.  Run: pip install httpx")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("demo")

STATION_IDS = ["DL-001", "DL-002", "DL-003", "DL-004", "DL-005"]
STATION_NAMES = {
    "DL-001": "Karol Bagh",
    "DL-002": "Sadar Bazar",
    "DL-003": "Connaught Place",
    "DL-004": "Lajpat Nagar",
    "DL-005": "Rohini Sector 9",
}

# Per-station baseline multiplier (some booths busier than others)
_MULTIPLIERS = {
    "DL-001": 1.10, "DL-002": 0.85, "DL-003": 1.25,
    "DL-004": 0.90, "DL-005": 0.75,
}


def _synthetic_count(hour: float, station_id: str) -> int:
    """Double-Gaussian crowd model: peak at 10am, 4pm; trough at 1pm."""
    mult = _MULTIPLIERS.get(station_id, 1.0)
    morning   = 80 * math.exp(-((hour - 10.0) ** 2) / 5.0)
    afternoon = 60 * math.exp(-((hour - 16.0) ** 2) / 3.0)
    noise     = random.gauss(0, 6)
    return max(0, int((morning + afternoon + noise) * mult))


def _crowd_emoji(level: str) -> str:
    return {"short": "🟢", "moderate": "🟡", "long": "🔴"}.get(level, "⚪")


async def _post_update(client: httpx.AsyncClient, api_base: str, station_id: str) -> dict | None:
    hour  = time.localtime().tm_hour + time.localtime().tm_min / 60.0
    count = _synthetic_count(hour, station_id)
    queue_m = count * 0.55

    url = f"{api_base}/stations/{station_id}/update"
    try:
        r = await client.post(
            url,
            json={"count": float(count), "queue_length_m": queue_m},
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        log.warning("Cannot connect to %s  — is the server running?", api_base)
        return None
    except Exception as exc:
        log.warning("POST failed for %s: %s", station_id, exc)
        return None


def _print_voter_view(station_id: str, data: dict) -> None:
    name  = STATION_NAMES.get(station_id, station_id)
    level = data.get("crowd_level", "unknown")
    wait  = data.get("wait_expected_sec", 0)
    count = data.get("person_count", 0)
    q_len = data.get("queue_length_m", 0)
    emoji = _crowd_emoji(level)
    wait_min = round(wait / 60)

    print(f"\n  {emoji}  {name}  ({station_id})")
    print(f"     {count:.0f} people in queue  ·  {q_len:.0f} m long")
    print(f"     Expected wait: ~{wait_min} min  [{level.upper()}]")


async def main(api_base: str, interval: float, once: bool) -> None:
    print(f"\n{'='*60}")
    print(f"  QCI Demo — Delhi Election Day Simulation")
    print(f"  API: {api_base}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient() as client:
        cycle = 0
        while True:
            if once:
                # Post once to every station
                for sid in STATION_IDS:
                    data = await _post_update(client, api_base, sid)
                    if data:
                        _print_voter_view(sid, data)
                print("\n[demo --once] Seeded all stations. Exiting.\n")
                return
            else:
                # Rotate through stations deterministically
                sid  = STATION_IDS[cycle % len(STATION_IDS)]
                data = await _post_update(client, api_base, sid)
                if data:
                    _print_voter_view(sid, data)
                cycle += 1
                log.info("Sleeping %gs …", interval)
                await asyncio.sleep(interval)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QCI live demo simulator.")
    p.add_argument("--api",      default="http://localhost:8000", help="API base URL.")
    p.add_argument("--interval", type=float, default=30.0, help="Seconds between posts.")
    p.add_argument("--once",     action="store_true", help="Post once to every station then exit.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(main(args.api, args.interval, args.once))
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")
