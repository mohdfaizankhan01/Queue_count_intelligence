"""OpenCV HOG-based people detector (Layer 2 baseline, extended in Layer 3)."""

from __future__ import annotations

import numpy as np
import torch

from .base import CrowdCounter


class HOGCounter(CrowdCounter):
    """People counter using OpenCV's default HOG + SVM detector.

    Provides both ``forward()`` (counts only, compatible with CrowdCounter
    interface) and ``detect()`` (counts + xyxy bounding boxes).
    """

    def __init__(
        self,
        win_stride: tuple[int, int] = (8, 8),
        scale: float = 1.05,
        padding: tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__()
        try:
            import cv2
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._win_stride = win_stride
            self._scale = scale
            self._padding = padding
            self._cv2 = cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for HOGCounter") from exc

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return iter([])

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _detect_single(self, img_np: np.ndarray) -> tuple[float, np.ndarray]:
        """Run HOG on one (H,W,C) uint8 image.  Returns (count, xyxy_boxes)."""
        if img_np.shape[2] == 3:
            img_gray = self._cv2.cvtColor(img_np, self._cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np[:, :, 0]

        h, w = img_gray.shape
        if h < 128 or w < 64:
            img_gray = self._cv2.resize(img_gray, (max(w, 64), max(h, 128)))

        rects, _ = self._hog.detectMultiScale(
            img_gray,
            winStride=self._win_stride,
            padding=self._padding,
            scale=self._scale,
        )

        if len(rects) > 0:
            # rects are (x, y, w, h) — convert to xyxy
            boxes = np.array(
                [[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in rects],
                dtype=np.float32,
            )
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        return float(len(boxes)), boxes

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[np.ndarray]]:
        """Detect persons in a batch of images.

        Args:
            x: ``(B, C, H, W)`` float32 tensor in ``[0, 1]``.

        Returns:
            counts:  ``(B,)`` float32 tensor.
            boxes:   list of length B; each element is ``(N, 4)`` float32
                     bounding-box array in xyxy pixel format.
        """
        counts: list[float] = []
        all_boxes: list[np.ndarray] = []

        for i in range(x.shape[0]):
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cnt, boxes = self._detect_single(img_np)
            counts.append(cnt)
            all_boxes.append(boxes)

        return torch.tensor(counts, dtype=torch.float32), all_boxes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B,)`` person count tensor."""
        counts, _ = self.detect(x)
        return counts
