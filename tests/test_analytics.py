"""Tests for Layer 5: ServiceRateModel, QueueStatus, HistoryTracker."""

import pytest

from qci.analytics import ServiceRateModel, QueueStatus, HistoryTracker, crowd_level_from_wait


# ---------------------------------------------------------------------------
# ServiceRateModel
# ---------------------------------------------------------------------------

class TestServiceRateModel:
    def test_wait_zero_queue(self):
        m = ServiceRateModel(n_booths=3, avg_service_time_sec=120)
        w = m.estimate_wait(0)
        # With no queue, wait = service time
        assert w.expected_sec == pytest.approx(120.0)

    def test_wait_monotone_in_queue(self):
        m = ServiceRateModel(n_booths=3, avg_service_time_sec=120)
        waits = [m.estimate_wait(q).expected_sec for q in [0, 5, 10, 20, 50]]
        assert all(waits[i] <= waits[i + 1] for i in range(len(waits) - 1))

    def test_pessimistic_gt_expected_gt_optimistic(self):
        m = ServiceRateModel(n_booths=3, avg_service_time_sec=120)
        for q in [0, 10, 30]:
            w = m.estimate_wait(q)
            assert w.pessimistic_sec >= w.expected_sec >= w.optimistic_sec

    def test_n_booths_reduces_wait(self):
        q = 30
        w1 = ServiceRateModel(n_booths=1).estimate_wait(q)
        w3 = ServiceRateModel(n_booths=3).estimate_wait(q)
        assert w1.expected_sec > w3.expected_sec

    def test_erlang_c_in_range(self):
        m = ServiceRateModel(n_booths=3)
        for q in [0, 5, 20]:
            w = m.estimate_wait(q)
            if w.erlang_c_prob is not None:
                assert 0.0 <= w.erlang_c_prob <= 1.0

    def test_invalid_booths(self):
        with pytest.raises(ValueError):
            ServiceRateModel(n_booths=0)

    def test_invalid_service_time(self):
        with pytest.raises(ValueError):
            ServiceRateModel(avg_service_time_sec=-5)


# ---------------------------------------------------------------------------
# crowd_level_from_wait
# ---------------------------------------------------------------------------

class TestCrowdLevel:
    def test_short(self):
        assert crowd_level_from_wait(0) == "short"
        assert crowd_level_from_wait(9 * 60) == "short"

    def test_moderate(self):
        assert crowd_level_from_wait(10 * 60) == "moderate"
        assert crowd_level_from_wait(20 * 60) == "moderate"
        assert crowd_level_from_wait(25 * 60) == "moderate"

    def test_long(self):
        assert crowd_level_from_wait(26 * 60) == "long"
        assert crowd_level_from_wait(60 * 60) == "long"


# ---------------------------------------------------------------------------
# QueueStatus
# ---------------------------------------------------------------------------

class TestQueueStatus:
    def _make_status(self, q: float = 10, n_booths: int = 3) -> QueueStatus:
        model = ServiceRateModel(n_booths=n_booths)
        wait = model.estimate_wait(q)
        return QueueStatus.create(
            station_id="TEST_01",
            person_count=q,
            queue_length_m=q * 0.6,
            crowd_density=0.5,
            wait_estimate=wait,
        )

    def test_crowd_level_consistent(self):
        s = self._make_status(q=100)
        assert s.crowd_level in ("short", "moderate", "long")

    def test_json_roundtrip(self):
        s = self._make_status(q=25)
        s2 = QueueStatus.from_json(s.to_json())
        assert s2.station_id == s.station_id
        assert s2.person_count == pytest.approx(s.person_count)
        assert s2.crowd_level == s.crowd_level

    def test_summary_contains_station_id(self):
        s = self._make_status()
        assert "TEST_01" in s.summary()

    def test_wait_expected_min(self):
        s = self._make_status(q=5)
        assert s.wait_expected_min == pytest.approx(s.wait_expected_sec / 60.0)


# ---------------------------------------------------------------------------
# HistoryTracker
# ---------------------------------------------------------------------------

class TestHistoryTracker:
    def _make_status(self, station_id: str, q: float = 10) -> QueueStatus:
        model = ServiceRateModel(n_booths=3)
        wait = model.estimate_wait(q)
        return QueueStatus.create(
            station_id=station_id,
            person_count=q,
            queue_length_m=q * 0.5,
            crowd_density=0.4,
            wait_estimate=wait,
        )

    def test_insert_and_latest(self):
        tracker = HistoryTracker(":memory:")
        s = self._make_status("S1", q=20)
        tracker.insert(s)
        latest = tracker.get_latest("S1")
        assert latest is not None
        assert latest.station_id == "S1"
        assert latest.person_count == pytest.approx(20.0)

    def test_latest_returns_none_for_unknown_station(self):
        tracker = HistoryTracker(":memory:")
        assert tracker.get_latest("UNKNOWN") is None

    def test_history_window(self):
        tracker = HistoryTracker(":memory:")
        for q in [10, 20, 30]:
            tracker.insert(self._make_status("S2", q=q))
        history = tracker.get_history("S2", hours=24)
        assert len(history) == 3

    def test_history_returns_correct_station(self):
        tracker = HistoryTracker(":memory:")
        tracker.insert(self._make_status("S3", q=5))
        tracker.insert(self._make_status("S4", q=50))
        rows = tracker.get_history("S3", hours=24)
        assert all(r.station_id == "S3" for r in rows)

    def test_all_stations_latest(self):
        tracker = HistoryTracker(":memory:")
        for sid in ["A", "B", "C"]:
            tracker.insert(self._make_status(sid, q=15))
        all_latest = tracker.get_all_stations_latest()
        station_ids = {r.station_id for r in all_latest}
        assert station_ids == {"A", "B", "C"}

    def test_latest_reflects_most_recent(self):
        tracker = HistoryTracker(":memory:")
        tracker.insert(self._make_status("S5", q=10))
        tracker.insert(self._make_status("S5", q=99))
        latest = tracker.get_latest("S5")
        assert latest.person_count == pytest.approx(99.0)

    def test_close_and_reuse(self):
        tracker = HistoryTracker(":memory:")
        tracker.insert(self._make_status("S6"))
        tracker.close()
        # After close further queries raise; just test close doesn't crash
