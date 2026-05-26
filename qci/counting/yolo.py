"""YOLOv8 person counter.

Requires ``pip install ultralytics``.  If the package is absent the counter
logs a warning and returns 0 counts so the rest of the pipeline (including
CI tests that do not install ultralytics) continues without breaking.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import torch

from .base import CrowdCounter

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0   # COCO class id for "person"


class YOLOCounter(CrowdCounter):
    """YOLOv8-based person counter.

    Filters COCO detections to class 0 (person) above *confidence_threshold*.
    Provides both ``forward()`` (counts only) and ``detect()`` (counts + boxes).
    Falls back gracefully when ultralytics is not installed.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self._available = False
        self._model = None

        try:
            from ultralytics import YOLO
            self._model = YOLO(model_name)
            if device:
                self._model.to(device)
            self._available = True
            logger.info(f"YOLOCounter: loaded {model_name}")
        except ImportError:
            warnings.warn(
                "ultralytics is not installed; YOLOCounter will return 0 for all images. "
                "Install with: pip install ultralytics"
            )
        except Exception as exc:
            warnings.warn(f"YOLOCounter: could not load {model_name} ({exc}); returning 0.")

    # YOLOv8 is not an nn.Module we own — register no parameters
    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return iter([])

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[np.ndarray]]:
        """Detect persons in a batch of images.

        Args:
            x: ``(B, C, H, W)`` float32 tensor in ``[0, 1]``.

        Returns:
            counts:    ``(B,)`` float32 tensor of person counts.
            boxes:     list of length B, each element an (N, 4) float32 array
                       of bounding boxes in xyxy pixel format.
        """
        counts: list[float] = []
        all_boxes: list[np.ndarray] = []

        if not self._available:
            empty = np.zeros((0, 4), dtype=np.float32)
            return torch.zeros(x.shape[0], dtype=torch.float32), [empty] * x.shape[0]

        for i in range(x.shape[0]):
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            results = self._model(
                img_np,
                conf=self.confidence_threshold,
                classes=[_PERSON_CLASS_ID],
                verbose=False,
            )
            boxes_tensor = results[0].boxes.xyxy  # (N, 4) on whatever device
            if len(boxes_tensor) > 0:
                boxes_np = boxes_tensor.cpu().numpy().astype(np.float32)
            else:
                boxes_np = np.zeros((0, 4), dtype=np.float32)
            counts.append(float(len(boxes_np)))
            all_boxes.append(boxes_np)

        return torch.tensor(counts, dtype=torch.float32), all_boxes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B,)`` person count tensor."""
        counts, _ = self.detect(x)
        return counts
