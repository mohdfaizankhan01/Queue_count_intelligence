"""Tests for Layer 7: FastAPI server (api.py) and seed/worker utilities."""

import json
import os

import pytest

# Use in-memory DB so tests don't write to disk
os.environ.setdefault("DB_PATH", ":memory:")

from fastapi.testclient import TestClient

from qci.server.api import app
from qci.server.config import reset_config
from qci.server.seed import generate_history
from qci.server.stations import DELHI_STATIONS, get_station_meta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient that runs the full app lifespan (DB init, etc.)."""
    reset_config()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /stations
# ---------------------------------------------------------------------------

class TestListStations:
    def test_returns_200(self, client):
        r = client.get("/stations")
        assert r.status_code == 200

    def test_response_has_stations_key(self, client):
        data = client.get("/stations").json()
        assert "stations" in data

    def test_returns_five_delhi_stations(self, client):
        data = client.get("/stations").json()
        ids = {s["id"] for s in data["stations"]}
        assert {"DL-001", "DL-002", "DL-003", "DL-004", "DL-005"} == ids

    def test_station_has_required_fields(self, client):
        data = client.get("/stations").json()
        for s in data["stations"]:
            assert "id" in s
            assert "name" in s
            assert "lat" in s
            assert "lon" in s
            # latest may be None if no data
            assert "latest" in s


# ---------------------------------------------------------------------------
# GET /stations/{station_id}
# ---------------------------------------------------------------------------

class TestGetStation:
    def test_returns_200_for_known_station(self, client):
        r = client.get("/stations/DL-001")
        assert r.status_code == 200

    def test_response_has_history_key(self, client):
        data = client.get("/stations/DL-001").json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_unknown_station_returns_empty_history(self, client):
        data = client.get("/stations/UNKNOWN-99").json()
        assert data["history"] == []
        assert data["latest"] is None


# ---------------------------------------------------------------------------
# POST /stations/{station_id}/update (JSON body)
# ---------------------------------------------------------------------------

class TestUpdateStation:
    def test_json_update_returns_200(self, client):
        r = client.post(
            "/stations/DL-001/update",
            json={"count": 25.0, "queue_length_m": 12.5},
        )
        assert r.status_code == 200

    def test_response_has_queue_status_fields(self, client):
        r = client.post(
            "/stations/DL-002/update",
            json={"count": 40.0, "queue_length_m": 20.0},
        )
        data = r.json()
        assert data["station_id"] == "DL-002"
        assert data["person_count"] == pytest.approx(40.0)
        assert data["crowd_level"] in ("short", "moderate", "long")
        assert "wait_expected_sec" in data

    def test_update_appears_in_latest(self, client):
        client.post(
            "/stations/DL-003/update",
            json={"count": 55.0, "queue_length_m": 28.0},
        )
        detail = client.get("/stations/DL-003").json()
        assert detail["latest"] is not None
        assert detail["latest"]["person_count"] == pytest.approx(55.0)

    def test_zero_count_accepted(self, client):
        r = client.post(
            "/stations/DL-004/update",
            json={"count": 0, "queue_length_m": 0},
        )
        assert r.status_code == 200
        assert r.json()["crowd_level"] == "short"

    def test_large_count_gives_long_crowd(self, client):
        r = client.post(
            "/stations/DL-005/update",
            json={"count": 200.0, "queue_length_m": 100.0},
        )
        assert r.json()["crowd_level"] == "long"


# ---------------------------------------------------------------------------
# GET /stations/{station_id}/privacy_check
# ---------------------------------------------------------------------------

class TestPrivacyCheck:
    def test_returns_200(self, client):
        r = client.get("/stations/DL-001/privacy_check")
        assert r.status_code == 200

    def test_has_encoding_strength(self, client):
        data = client.get("/stations/DL-001/privacy_check").json()
        assert "encoding_strength" in data
        assert 0.0 <= data["encoding_strength"] <= 1.0

    def test_has_station_id(self, client):
        data = client.get("/stations/DL-002/privacy_check").json()
        assert data["station_id"] == "DL-002"

    def test_has_privacy_protected_field(self, client):
        data = client.get("/stations/DL-001/privacy_check").json()
        # May be None if no privacy CSV; bool otherwise
        assert "privacy_protected" in data


# ---------------------------------------------------------------------------
# Root / frontend
# ---------------------------------------------------------------------------

class TestFrontend:
    def test_root_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

class TestSeed:
    def test_generate_history_returns_records(self):
        records = generate_history("DL-001", hours=1.0, interval_minutes=30)
        assert len(records) >= 2

    def test_generate_history_station_id_correct(self):
        records = generate_history("DL-002", hours=0.5)
        assert all(r.station_id == "DL-002" for r in records)

    def test_generate_history_crowd_level_valid(self):
        records = generate_history("DL-003", hours=1.0, interval_minutes=30)
        for r in records:
            assert r.crowd_level in ("short", "moderate", "long")

    def test_generate_history_person_count_non_negative(self):
        records = generate_history("DL-001", hours=3.0, interval_minutes=15)
        assert all(r.person_count >= 0 for r in records)


# ---------------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------------

class TestStationMeta:
    def test_known_station_has_name(self):
        meta = get_station_meta("DL-001")
        assert meta["name"] == "Karol Bagh Primary School"

    def test_unknown_station_fallback(self):
        meta = get_station_meta("UNKNOWN")
        assert meta["id"] == "UNKNOWN"

    def test_all_stations_have_coords(self):
        for s in DELHI_STATIONS:
            assert s["lat"] is not None
            assert s["lon"] is not None
            assert 28.0 < s["lat"] < 29.0  # Delhi latitude range
            assert 76.5 < s["lon"] < 78.0  # Delhi longitude range
