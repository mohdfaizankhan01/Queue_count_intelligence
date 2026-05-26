"""Queuing-theory wait-time estimator for a polling station.

Implements a simplified M/M/c queueing model.  For c servers each with
service rate μ = 1/avg_service_time_sec the expected wait is:

    If queue ≤ c:  W = avg_service_time_sec   (no queue, direct service)
    Otherwise:     W ≈ (queue - c) / (c·μ) + avg_service_time_sec
                         ────────────────────   ────────────────────
                         time to clear excess    own service time

Confidence bounds (±20 % on service time) give optimistic / pessimistic
estimates.

The Erlang-C probability (exact M/M/c formula) is also computed and returned
as ``erlang_c_prob`` for reference, but the simplified formula above is used
for wait estimation to avoid numerical edge cases near saturation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class WaitEstimate:
    optimistic_sec: float
    expected_sec: float
    pessimistic_sec: float
    erlang_c_prob: Optional[float] = None   # P(customer must wait)

    @property
    def optimistic_min(self) -> float:
        return self.optimistic_sec / 60.0

    @property
    def expected_min(self) -> float:
        return self.expected_sec / 60.0

    @property
    def pessimistic_min(self) -> float:
        return self.pessimistic_sec / 60.0


def _erlang_c(c: int, rho_total: float) -> float:
    """Erlang-C probability P(W > 0) for M/M/c queue.

    Args:
        c:          Number of servers.
        rho_total:  Total offered load λ/μ  (must be < c for stability).

    Returns:
        Probability that an arriving customer has to wait.
    """
    if rho_total >= c:
        return 1.0   # saturated — all customers wait

    rho = rho_total / c   # per-server utilisation

    # Σ_{k=0}^{c-1} (c·ρ)^k / k!
    sum_term = sum((rho_total ** k) / math.factorial(k) for k in range(c))
    last_term = (rho_total ** c) / math.factorial(c) / (1.0 - rho)
    return last_term / (sum_term + last_term)


class ServiceRateModel:
    """M/M/c queuing model for a polling station.

    Parameters
    ----------
    n_booths:            Number of active voting booths.
    avg_service_time_sec: Mean time to serve one voter (default 120 s = 2 min).
    confidence_pct:      Fractional width of confidence interval on service time
                         (default 0.20 → ±20 %).
    """

    def __init__(
        self,
        n_booths: int = 3,
        avg_service_time_sec: float = 120.0,
        confidence_pct: float = 0.20,
    ) -> None:
        if n_booths < 1:
            raise ValueError("n_booths must be >= 1")
        if avg_service_time_sec <= 0:
            raise ValueError("avg_service_time_sec must be positive")
        self.n_booths = n_booths
        self.avg_service_time_sec = avg_service_time_sec
        self.confidence_pct = confidence_pct

    # ------------------------------------------------------------------

    def _single_wait(self, queue: float, service_time: float) -> float:
        """Simple M/M/c wait for a given service time."""
        c = self.n_booths
        mu = 1.0 / service_time
        if queue <= c:
            # No actual queue — wait is just the service time
            return service_time
        # Simplified: time to drain excess queue + own service time
        return (queue - c) / (c * mu) + service_time

    def estimate_wait(self, queue_length_persons: float) -> WaitEstimate:
        """Estimate wait time for a queue of *queue_length_persons* people.

        Returns optimistic (fast service), expected, and pessimistic (slow
        service) estimates, each in seconds.
        """
        q = max(0.0, float(queue_length_persons))
        t_exp = self._single_wait(q, self.avg_service_time_sec)
        t_opt = self._single_wait(q, self.avg_service_time_sec * (1.0 - self.confidence_pct))
        t_pes = self._single_wait(q, self.avg_service_time_sec * (1.0 + self.confidence_pct))

        # Erlang-C (informational)
        mu = 1.0 / self.avg_service_time_sec
        # Rough arrival rate: assume queue represents ~5-min accumulation
        lam = q / max(t_exp, 1.0) if t_exp > 0 else 0.0
        rho_total = lam / mu if mu > 0 else 0.0
        try:
            ec = _erlang_c(self.n_booths, rho_total)
        except (OverflowError, ZeroDivisionError):
            ec = None

        return WaitEstimate(
            optimistic_sec=t_opt,
            expected_sec=t_exp,
            pessimistic_sec=t_pes,
            erlang_c_prob=ec,
        )
