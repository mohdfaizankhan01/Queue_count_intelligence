"""Crowd-counting evaluation metrics: MAE, RMSE, and GAME(L)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------


def absolute_error(pred: float, gt: float) -> float:
    """Absolute error for a single image."""
    return abs(pred - gt)


def squared_error(pred: float, gt: float) -> float:
    """Squared error for a single image."""
    return (pred - gt) ** 2


def game(
    pred_dmap: np.ndarray,
    gt_dmap: np.ndarray,
    level: int = 0,
) -> float:
    """Grid Average Mean absolute Error (GAME) for a single image.

    GAME(L) divides the image into 4^L non-overlapping regions and sums the
    absolute density-count difference over all regions.  GAME(0) equals the
    plain absolute count error.

    Args:
        pred_dmap: ``(H, W)`` predicted density map.
        gt_dmap:   ``(H, W)`` ground-truth density map.
        level:     L ≥ 0.

    Returns:
        Scalar GAME error for this image.
    """
    if level < 0:
        raise ValueError("GAME level must be >= 0")

    n = 2 ** level  # patches per spatial dimension
    H, W = pred_dmap.shape

    total = 0.0
    for i in range(n):
        for j in range(n):
            r0 = i * H // n
            r1 = (i + 1) * H // n if i < n - 1 else H
            c0 = j * W // n
            c1 = (j + 1) * W // n if j < n - 1 else W
            total += abs(
                float(pred_dmap[r0:r1, c0:c1].sum())
                - float(gt_dmap[r0:r1, c0:c1].sum())
            )
    return total


# ---------------------------------------------------------------------------
# Aggregate over a list of per-image errors
# ---------------------------------------------------------------------------


def compute_mae(errors: Sequence[float]) -> float:
    """Mean Absolute Error over a list of per-image absolute errors."""
    return float(np.mean(errors)) if errors else float("nan")


def compute_rmse(sq_errors: Sequence[float]) -> float:
    """Root Mean Squared Error over a list of per-image squared errors."""
    return float(math.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")


def compute_game(game_errors: Sequence[float]) -> float:
    """Mean GAME error over a list of per-image GAME values."""
    return float(np.mean(game_errors)) if game_errors else float("nan")


def aggregate_metrics(records: list[dict]) -> dict[str, float]:
    """Compute MAE, RMSE, and GAME(0..3) from a list of per-image dicts.

    Each dict must have keys: ``pred``, ``gt``, and optionally
    ``game_0`` … ``game_3``.
    """
    aes = [absolute_error(r["pred"], r["gt"]) for r in records]
    ses = [squared_error(r["pred"], r["gt"]) for r in records]
    out = {"mae": compute_mae(aes), "rmse": compute_rmse(ses)}
    for lvl in range(4):
        key = f"game_{lvl}"
        vals = [r[key] for r in records if key in r]
        if vals:
            out[key] = compute_game(vals)
    return out
