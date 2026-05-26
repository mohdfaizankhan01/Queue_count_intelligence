"""HistoryTracker — lightweight SQLite store for QueueStatus records."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from .queue_status import QueueStatus


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS queue_status (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id       TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,
    person_count     REAL,
    queue_length_m   REAL,
    crowd_density    REAL,
    wait_optimistic  REAL,
    wait_expected    REAL,
    wait_pessimistic REAL,
    crowd_level      TEXT,
    encoding_strength REAL,
    notes            TEXT,
    inserted_at      REAL    NOT NULL    -- Unix time for ordering
);
"""

_INSERT = """
INSERT INTO queue_status
  (station_id, timestamp, person_count, queue_length_m, crowd_density,
   wait_optimistic, wait_expected, wait_pessimistic, crowd_level,
   encoding_strength, notes, inserted_at)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?);
"""

_LATEST = """
SELECT * FROM queue_status
WHERE station_id = ?
ORDER BY inserted_at DESC
LIMIT 1;
"""

_HISTORY = """
SELECT * FROM queue_status
WHERE station_id = ?
  AND inserted_at >= ?
ORDER BY inserted_at ASC;
"""

_ALL_LATEST = """
SELECT q.*
FROM queue_status q
INNER JOIN (
    SELECT station_id, MAX(inserted_at) AS mx
    FROM queue_status
    GROUP BY station_id
) t ON q.station_id = t.station_id AND q.inserted_at = t.mx;
"""


class HistoryTracker:
    """Persist and query ``QueueStatus`` records in a local SQLite database.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Defaults to ``results/history.db``.
        Pass ``":memory:"`` for an in-memory test database.
    """

    def __init__(self, db_path: str | Path = "results/history.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------

    def insert(self, status: QueueStatus) -> None:
        """Persist a QueueStatus snapshot."""
        self._conn.execute(
            _INSERT,
            (
                status.station_id,
                status.timestamp,
                status.person_count,
                status.queue_length_m,
                status.crowd_density,
                status.wait_optimistic_sec,
                status.wait_expected_sec,
                status.wait_pessimistic_sec,
                status.crowd_level,
                status.encoding_strength,
                status.notes,
                time.time(),
            ),
        )
        self._conn.commit()

    def get_latest(self, station_id: str) -> Optional[QueueStatus]:
        """Return the most recent record for *station_id*, or None."""
        row = self._conn.execute(_LATEST, (station_id,)).fetchone()
        return self._row_to_status(row) if row else None

    def get_history(self, station_id: str, hours: float = 2.0) -> List[QueueStatus]:
        """Return records for *station_id* from the past *hours* hours."""
        since = time.time() - hours * 3600
        rows = self._conn.execute(_HISTORY, (station_id, since)).fetchall()
        return [self._row_to_status(r) for r in rows]

    def get_all_stations_latest(self) -> List[QueueStatus]:
        """Return the most recent record for every known station."""
        rows = self._conn.execute(_ALL_LATEST).fetchall()
        return [self._row_to_status(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_status(row: sqlite3.Row) -> QueueStatus:
        return QueueStatus(
            station_id=row["station_id"],
            timestamp=row["timestamp"],
            person_count=row["person_count"],
            queue_length_m=row["queue_length_m"],
            crowd_density=row["crowd_density"],
            wait_optimistic_sec=row["wait_optimistic"],
            wait_expected_sec=row["wait_expected"],
            wait_pessimistic_sec=row["wait_pessimistic"],
            crowd_level=row["crowd_level"],
            encoding_strength=row["encoding_strength"] or 0.0,
            notes=row["notes"] or "",
        )
