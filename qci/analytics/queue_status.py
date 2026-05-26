"""QueueStatus dataclass — snapshot of one polling station at one moment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional, Tuple

CrowdLevel = Literal["short", "moderate", "long"]

_SHORT_THRESHOLD_MIN = 10.0
_LONG_THRESHOLD_MIN = 25.0


def crowd_level_from_wait(expected_wait_sec: float) -> CrowdLevel:
    """Classify wait into crowd_level string.

    * ``"short"``    — expected wait < 10 min
    * ``"moderate"`` — 10 – 25 min
    * ``"long"``     — > 25 min
    """
    minutes = expected_wait_sec / 60.0
    if minutes < _SHORT_THRESHOLD_MIN:
        return "short"
    elif minutes <= _LONG_THRESHOLD_MIN:
        return "moderate"
    else:
        return "long"


@dataclass
class QueueStatus:
    """Complete snapshot of a station at a single point in time.

    All times are in seconds.  Serialisable to / from JSON for the API layer
    (Layer 7).
    """

    station_id: str
    timestamp: str                           # ISO-8601
    person_count: float
    queue_length_m: float
    crowd_density: float                     # persons / m²
    wait_optimistic_sec: float
    wait_expected_sec: float
    wait_pessimistic_sec: float
    crowd_level: CrowdLevel
    encoding_strength: float = 0.0           # for research logging
    notes: str = ""

    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        station_id: str,
        person_count: float,
        queue_length_m: float,
        crowd_density: float,
        wait_estimate,                       # WaitEstimate from ServiceRateModel
        encoding_strength: float = 0.0,
        notes: str = "",
        timestamp: Optional[str] = None,
    ) -> "QueueStatus":
        """Convenience constructor that derives crowd_level automatically."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        level = crowd_level_from_wait(wait_estimate.expected_sec)
        return cls(
            station_id=station_id,
            timestamp=ts,
            person_count=person_count,
            queue_length_m=queue_length_m,
            crowd_density=crowd_density,
            wait_optimistic_sec=wait_estimate.optimistic_sec,
            wait_expected_sec=wait_estimate.expected_sec,
            wait_pessimistic_sec=wait_estimate.pessimistic_sec,
            crowd_level=level,
            encoding_strength=encoding_strength,
            notes=notes,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "QueueStatus":
        d = json.loads(s)
        return cls(**d)

    @property
    def wait_expected_min(self) -> float:
        return self.wait_expected_sec / 60.0

    def summary(self) -> str:
        return (
            f"[{self.station_id}] {self.timestamp} | "
            f"persons={self.person_count:.0f} | "
            f"queue={self.queue_length_m:.1f} m | "
            f"wait≈{self.wait_expected_min:.1f} min ({self.crowd_level})"
        )
