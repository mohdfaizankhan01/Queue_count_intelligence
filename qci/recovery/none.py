"""Identity (no-op) restoration."""

from __future__ import annotations

import torch

from .base import RestorationModule


class IdentityRestoration(RestorationModule):
    """Pass-through: returns the measurement unchanged."""

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return y
