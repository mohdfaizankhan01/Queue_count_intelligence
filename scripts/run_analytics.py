#!/usr/bin/env python3
"""run_analytics.py — record a queue snapshot and print an ETA report.

Usage examples
--------------
Record a manual observation::

    python scripts/run_analytics.py \\
        --station_id ABC \\
        --count 47 \\
        --queue_length_m 23

Query the recent history for a station::

    python scripts/run_analytics.py --station_id ABC --history

Show the latest snapshot for every known station::

    python scripts/run_analytics.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from qci.analytics import ServiceRateModel, QueueStatus, HistoryTracker


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record and query queue analytics for a polling station."
    )
    p.add_argument("--station_id", default="STATION_A", help="Station identifier")
    p.add_argument("--count", type=float, default=None, help="Observed person count")
    p.add_argument("--queue_length_m", type=float, default=None, help="Queue length in metres")
    p.add_argument("--crowd_density", type=float, default=None, help="Density (persons/m²)")
    p.add_argument("--n_booths", type=int, default=3, help="Active voting booths")
    p.add_argument("--service_time_sec", type=float, default=120.0, help="Mean service time (s)")
    p.add_argument("--db", default="results/history.db", help="SQLite database path")
    p.add_argument("--history", action="store_true", help="Print recent history instead of inserting")
    p.add_argument("--history_hours", type=float, default=2.0, help="Hours of history to show")
    p.add_argument("--all", dest="all_stations", action="store_true", help="Show latest for all stations")
    return p


def _print_eta_report(status: QueueStatus, model: ServiceRateModel) -> None:
    print("\n" + "=" * 60)
    print(f"  Station : {status.station_id}")
    print(f"  Time    : {status.timestamp}")
    print(f"  Persons : {status.person_count:.0f}")
    print(f"  Queue   : {status.queue_length_m:.1f} m")
    if status.crowd_density:
        print(f"  Density : {status.crowd_density:.2f} p/m²")
    print("-" * 60)
    w = model.estimate_wait(status.person_count)
    print(f"  Wait (optimistic)  : {w.optimistic_min:5.1f} min")
    print(f"  Wait (expected)    : {w.expected_min:5.1f} min  ← ETA")
    print(f"  Wait (pessimistic) : {w.pessimistic_min:5.1f} min")
    if w.erlang_c_prob is not None:
        print(f"  P(must wait)       : {w.erlang_c_prob:.1%}")
    print(f"  Crowd level        : {status.crowd_level.upper()}")
    print("=" * 60)


def main() -> None:
    args = _build_parser().parse_args()
    tracker = HistoryTracker(db_path=args.db)

    if args.all_stations:
        rows = tracker.get_all_stations_latest()
        if not rows:
            print("No records found.")
        for s in rows:
            print(s.summary())
        return

    if args.history:
        rows = tracker.get_history(args.station_id, hours=args.history_hours)
        if not rows:
            print(f"No history found for station '{args.station_id}'.")
        for s in rows:
            print(s.summary())
        return

    # Insert mode — require count and queue_length_m
    if args.count is None or args.queue_length_m is None:
        print("ERROR: --count and --queue_length_m are required to record a snapshot.")
        sys.exit(1)

    model = ServiceRateModel(
        n_booths=args.n_booths,
        avg_service_time_sec=args.service_time_sec,
    )
    wait = model.estimate_wait(args.count)
    density = args.crowd_density if args.crowd_density is not None else 0.0

    status = QueueStatus.create(
        station_id=args.station_id,
        person_count=args.count,
        queue_length_m=args.queue_length_m,
        crowd_density=density,
        wait_estimate=wait,
    )

    tracker.insert(status)
    _print_eta_report(status, model)
    print(f"\nRecord saved to: {args.db}")


if __name__ == "__main__":
    main()
