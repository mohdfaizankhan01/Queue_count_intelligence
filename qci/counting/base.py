"""CrowdCounter abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CrowdCounter(ABC, nn.Module):
    """Interface for crowd counting models.

    All implementations accept a ``(B, C, H, W)`` image tensor and return a
    ``(B,)`` float tensor of predicted counts.

    .. TODO Layer 3: add YOLO-based counter, learned end-to-end counter here.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict crowd count.

        Args:
            x: ``(B, C, H, W)`` image tensor in ``[0, 1]``.

        Returns:
            ``(B,)`` float tensor of predicted counts.
        """
