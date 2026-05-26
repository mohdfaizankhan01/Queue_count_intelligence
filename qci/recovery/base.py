"""RestorationModule abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class RestorationModule(ABC, nn.Module):
    """Interface for image restoration modules.

    All subclasses receive the encoded (and optionally degraded) measurement
    ``y`` and return a restored estimate ``x_hat`` of the same shape.
    """

    @abstractmethod
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Restore encoded measurement.

        Args:
            y: ``(B, C, H, W)`` encoded measurement in ``[0, 1]``.

        Returns:
            ``(B, C, H, W)`` restored image, clamped to ``[0, 1]``.
        """
