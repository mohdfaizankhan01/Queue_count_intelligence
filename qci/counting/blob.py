"""BlobCounter — counts foreground blobs in synthetic crowd images.

Detection strategy: per-channel local standard deviation (3×3 window),
using the MAXIMUM std across the three colour channels.

The ``SyntheticCrowdDataset`` generates images where:
  - Each ellipse has ONE solid (R, G, B) colour applied uniformly to all its
    pixels.  Inside the ellipse, every channel is constant → local std = 0
    in EVERY channel.
  - Background: per-pixel i.i.d. uniform noise [0, 0.3] + smooth gradient.
    Per-channel local std ≈ 22 in uint8 units for any small window.

Taking the maximum std across all three channels lets us classify a pixel as
"uniform" (inside an ellipse) only when ALL channels are simultaneously quiet:

  max(std_R, std_G, std_B) ≈ 0    →  ellipse interior
  max(std_R, std_G, std_B) ≈ 22   →  noisy background

The probability that background noise has per-channel std < 11 in all three
channels simultaneously is ≈ (P_single)³ ≈ 0.05%, making false positives
essentially zero — and a morphological open removes any that remain.

Why a 3×3 window?
  The smallest generated ellipses have semi-axis 3 px.  A 3×3 window fits
  entirely inside such an ellipse for the 3×3 central pixel neighbourhood
  (diagonal distance √2 < 3), giving a ~9-pixel "pure-uniform" core with
  std = 0.  A 7×7 window would extend beyond the 3-px radius for every
  single pixel in the ellipse, mixing in background noise and raising the
  estimated std above 0 even at strength = 0 → the detector never fires.

How encoding degrades the detector:
  At strength = 0: ellipse pixels are perfectly uniform → max-channel std = 0
    → detected cleanly → count ≈ GT → low MAE.
  At strength > 0: Gaussian blur mixes ellipse pixels with the noisy
    background → per-channel std inside the ellipse increases → the uniform
    "core" shrinks and eventually drops below min_blob_area → undercounting
    → MAE rises monotonically with encoding strength.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .base import CrowdCounter


class BlobCounter(CrowdCounter):
    """Max-channel local-std crowd counter for synthetic ellipse-head images.

    Parameters
    ----------
    window_size:
        Side length of the square local-std window (pixels).  Must be small
        enough to fit inside the smallest ellipse (semi-axis ≥ 3 px); 3 is
        the default.
    std_threshold:
        A pixel is classified as "uniform" (inside an ellipse) when the
        maximum per-channel local std is below this value.  Background noise
        gives per-channel std ≈ 22 in uint8 units; a threshold of 11 (half)
        gives essentially zero false positives with the 3-channel criterion.
    min_blob_area:
        Minimum connected-component area (pixels) to be counted.  The 3×3
        uniform core inside a 3-px radius ellipse contains ~9 pixels, so the
        default of 6 is conservative.
    """

    def __init__(
        self,
        window_size: int = 3,
        std_threshold: float = 11.0,
        min_blob_area: int = 6,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.min_blob_area = min_blob_area

    # ------------------------------------------------------------------

    def _count_single(self, img_float: np.ndarray) -> float:
        """Count uniform-colour blobs in a single (H, W, 3) float32 image."""
        uint8 = (img_float * 255).clip(0, 255).astype(np.uint8)

        k = (self.window_size, self.window_size)
        max_std = np.zeros(uint8.shape[:2], dtype=np.float32)

        for ch in range(3):
            plane = uint8[:, :, ch].astype(np.float32)
            mean = cv2.boxFilter(plane, cv2.CV_32F, k, normalize=True)
            sq_mean = cv2.boxFilter(plane * plane, cv2.CV_32F, k, normalize=True)
            var = np.maximum(sq_mean - mean * mean, 0.0)
            np.maximum(max_std, np.sqrt(var), out=max_std)

        # Pixels where every channel is still uniform → inside an ellipse
        mask = (max_std < self.std_threshold).astype(np.uint8) * 255

        # Morphological cleanup: open removes stray noise pixels; close fills
        # any tiny holes in the blob interior left by per-pixel variation.
        k_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_elem)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_elem)

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        count = sum(
            1
            for i in range(1, n_labels)
            if stats[i, cv2.CC_STAT_AREA] >= self.min_blob_area
        )
        return float(count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: (B, 3, H, W) float tensor in [0, 1]

        Returns
        -------
        counts: (B,) float tensor
        """
        counts = [
            self._count_single(img_t.permute(1, 2, 0).detach().cpu().numpy())
            for img_t in x
        ]
        return torch.tensor(counts, dtype=torch.float32)
