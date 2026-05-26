"""Seed the SQLite database with synthetic historical data for 5 Delhi stations.

Usage::

    python -m qci.server.seed
    python -m qci.server.seed --db results/history.db --hours 3
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from qci.analytics import HistoryTracker, QueueStatus, ServiceRateModel
from qci.server.stations import DELHI_STATIONS

log = logging.getLogger(__name__)

# Per-station multiplier: some booths are busier than others
_STATION_MULTIPLIERS = {
    "DL-001": 1.10,
    "DL-002": 0.85,
    "DL-003": 1.25,
    "DL-004": 0.90,
    "DL-005": 0.75,
}


def _crowd_count(hour: float, multiplier: float = 1.0, seed_offset: int = 0) -> float:
    """Double-Gaussian election-day crowd model.

    Morning peak ~10am, afternoon peak ~16:00, trough midday.
    """
    rng = random.Random(int(hour * 100) + seed_offset)
    morning = 80.0 * math.exp(-((hour - 10.0) ** 2) / 5.0)
    afternoon = 60.0 * math.exp(-((hour - 16.0) ** 2) / 3.0)
    noise = rng.gauss(0, 5)
    return max(0.0, (morning + afternoon + noise) * multiplier)


def generate_history(
    station_id: str,
    hours: float = 3.0,
    interval_minutes: int = 15,
    n_booths: int = 3,
    seed: int = 0,
) -> List[QueueStatus]:
    """Generate synthetic QueueStatus records covering *hours* hours ending now."""
    model = ServiceRateModel(n_booths=n_booths, avg_service_time_sec=120.0)
    multiplier = _STATION_MULTIPLIERS.get(station_id, 1.0)

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    interval = timedelta(minutes=interval_minutes)

    records: List[QueueStatus] = []
    t = start
    idx = 0
    while t <= now:
        hour = t.hour + t.minute / 60.0
        count = _crowd_count(hour, multiplier, seed_offset=seed + idx)
        queue_length_m = count * 0.55
        wait = model.estimate_wait(count)
        status = QueueStatus.create(
            station_id=station_id,
            person_count=count,
            queue_length_m=queue_length_m,
            crowd_density=count / max(queue_length_m * 1.5, 1.0),
            wait_estimate=wait,
            encoding_strength=0.4,
            timestamp=t.isoformat(),
        )
        records.append(status)
        t += interval
        idx += 1

    return records


def seed_database(db_path: str, hours: float = 3.0) -> None:
    """Seed *db_path* with synthetic history for all 5 Delhi stations."""
    tracker = HistoryTracker(db_path=db_path)
    total = 0
    for i, station in enumerate(DELHI_STATIONS):
        sid = station["id"]
        records = generate_history(sid, hours=hours, seed=i * 1000)
        for r in records:
            tracker.insert(r)
        log.info("  %s (%s): %d records seeded", sid, station["name"], len(records))
        total += len(records)
    tracker.close()
    log.info("Seed complete: %d records across %d stations → %s", total, len(DELHI_STATIONS), db_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Seed the QCI database with synthetic history.")
    p.add_argument("--db", default="results/history.db", help="SQLite database path.")
    p.add_argument("--hours", type=float, default=3.0, help="Hours of history to generate per station.")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args()
    seed_database(db_path=args.db, hours=args.hours)
