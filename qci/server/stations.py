"""Static metadata for the five fictional Delhi polling stations."""

from __future__ import annotations

from typing import Dict, List, TypedDict


class StationMeta(TypedDict):
    id: str
    name: str
    lat: float
    lon: float
    ward: str


DELHI_STATIONS: List[StationMeta] = [
    {
        "id": "DL-001",
        "name": "Karol Bagh Primary School",
        "lat": 28.6519,
        "lon": 77.1909,
        "ward": "Karol Bagh",
    },
    {
        "id": "DL-002",
        "name": "Sadar Bazar Community Hall",
        "lat": 28.6621,
        "lon": 77.2189,
        "ward": "Sadar Bazar",
    },
    {
        "id": "DL-003",
        "name": "Connaught Place Civic Centre",
        "lat": 28.6315,
        "lon": 77.2167,
        "ward": "Connaught Place",
    },
    {
        "id": "DL-004",
        "name": "Lajpat Nagar Model School",
        "lat": 28.5665,
        "lon": 77.2431,
        "ward": "Lajpat Nagar",
    },
    {
        "id": "DL-005",
        "name": "Rohini Sector 9 Govt School",
        "lat": 28.7358,
        "lon": 77.1059,
        "ward": "Rohini",
    },
]

DELHI_STATIONS_BY_ID: Dict[str, StationMeta] = {s["id"]: s for s in DELHI_STATIONS}


def get_station_meta(station_id: str) -> StationMeta:
    return DELHI_STATIONS_BY_ID.get(
        station_id,
        {"id": station_id, "name": station_id, "lat": None, "lon": None, "ward": ""},
    )
