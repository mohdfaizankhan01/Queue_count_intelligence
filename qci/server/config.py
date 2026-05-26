"""Server configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerConfig:
    encoding_strength: float = float(os.getenv("ENCODING_STRENGTH", "0.4"))
    n_booths_default: int = int(os.getenv("N_BOOTHS_DEFAULT", "3"))
    counter_type: str = os.getenv("COUNTER_TYPE", "hog")
    avg_service_time_sec: float = float(os.getenv("AVG_SERVICE_TIME_SEC", "120.0"))
    db_path: str = os.getenv("DB_PATH", "results/history.db")
    privacy_results_csv: str = os.getenv("PRIVACY_RESULTS_CSV", "results/privacy_results.csv")


_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    global _config
    if _config is None:
        _config = ServerConfig()
    return _config


def reset_config() -> None:
    """Force re-read of environment variables (useful in tests)."""
    global _config
    _config = None
