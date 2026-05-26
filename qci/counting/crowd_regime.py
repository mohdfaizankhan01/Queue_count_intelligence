"""Crowd regime classification by count magnitude."""

from __future__ import annotations

from typing import Literal

CrowdRegime = Literal["sparse", "medium", "dense"]

# Thresholds (inclusive upper bound for sparse, lower bound for dense)
_SPARSE_MAX = 20
_DENSE_MIN = 80


def crowd_regime(count: float) -> CrowdRegime:
    """Classify a crowd count into a qualitative regime.

    Returns:
        ``"sparse"``  — count <  20
        ``"medium"``  — count in [20, 80]
        ``"dense"``   — count >  80
    """
    if count < _SPARSE_MAX:
        return "sparse"
    elif count <= _DENSE_MIN:
        return "medium"
    else:
        return "dense"


REGIMES: tuple[CrowdRegime, ...] = ("sparse", "medium", "dense")
