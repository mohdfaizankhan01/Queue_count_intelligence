"""Tests for evaluation metrics."""

import numpy as np
import pytest

from qci.eval.metrics import (
    absolute_error,
    aggregate_metrics,
    compute_game,
    compute_mae,
    compute_rmse,
    game,
    squared_error,
)


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------


def test_absolute_error_perfect():
    assert absolute_error(42.0, 42.0) == 0.0


def test_absolute_error_value():
    assert abs(absolute_error(10.0, 7.0) - 3.0) < 1e-9


def test_squared_error_perfect():
    assert squared_error(5.0, 5.0) == 0.0


def test_squared_error_value():
    assert abs(squared_error(4.0, 7.0) - 9.0) < 1e-9


# ---------------------------------------------------------------------------
# GAME metric
# ---------------------------------------------------------------------------


def test_game_level0_equals_count_error():
    """GAME(0) should equal |sum(pred) - sum(gt)|."""
    rng = np.random.default_rng(0)
    pred = rng.random((64, 64)).astype(np.float32) * 10
    gt = rng.random((64, 64)).astype(np.float32) * 10
    expected = abs(pred.sum() - gt.sum())
    assert abs(game(pred, gt, level=0) - expected) < 1e-3


def test_game_perfect_zero():
    dmap = np.ones((64, 64), dtype=np.float32)
    assert game(dmap, dmap, level=0) < 1e-6
    assert game(dmap, dmap, level=2) < 1e-6


def test_game_level_increases_sensitivity():
    """Higher L should give at least as large an error as lower L for non-uniform maps."""
    rng = np.random.default_rng(1)
    pred = rng.random((64, 64)).astype(np.float32)
    gt = np.zeros((64, 64), dtype=np.float32)
    # For a non-trivial prediction vs zero, GAME generally increases with L
    # (not guaranteed by definition, but true for random inputs with same total)
    g0 = game(pred, gt, level=0)
    g1 = game(pred, gt, level=1)
    assert g1 >= g0 - 1e-3, f"GAME(1)={g1:.3f} unexpectedly less than GAME(0)={g0:.3f}"


def test_game_negative_level():
    with pytest.raises(ValueError):
        game(np.zeros((4, 4)), np.zeros((4, 4)), level=-1)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def test_compute_mae_zero():
    assert compute_mae([0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_compute_rmse_zero():
    assert compute_rmse([0.0, 0.0]) == pytest.approx(0.0)


def test_compute_mae_value():
    assert compute_mae([1.0, 3.0, 5.0]) == pytest.approx(3.0)


def test_compute_rmse_value():
    # sqrt(mean([4,9])) = sqrt(6.5)
    assert compute_rmse([4.0, 9.0]) == pytest.approx(np.sqrt(6.5))


def test_aggregate_metrics():
    records = [
        {"pred": 10.0, "gt": 10.0, "game_0": 0.0},
        {"pred": 14.0, "gt": 10.0, "game_0": 4.0},
    ]
    out = aggregate_metrics(records)
    assert out["mae"] == pytest.approx(2.0)
    assert out["rmse"] == pytest.approx(np.sqrt((0 + 16) / 2))
    assert out["game_0"] == pytest.approx(2.0)


def test_empty_inputs():
    assert np.isnan(compute_mae([]))
    assert np.isnan(compute_rmse([]))
    assert np.isnan(compute_game([]))
