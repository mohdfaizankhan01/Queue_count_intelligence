"""Monocular depth estimation via Intel MiDaS (optional).

This module wraps the MiDaS "MiDaS_small" model (~100 MB) loaded via
``torch.hub``.  It is **optional**; when ``enabled=False`` in config the
pipeline falls back to the homography-only ground-plane projection.

Metric depth fusion
-------------------
MiDaS outputs relative inverse depth (disparity) d_rel where larger values
correspond to closer objects.  We calibrate to metric depth using the camera
ground-plane constraint:

    The pixel at the principal point (cx, cy) for a camera of height h and
    tilt θ should see a ground patch at metric depth:

        d_ground = h / sin(θ)   [metres along optical axis]

    Scale: s = d_ground / d_rel[cy, cx]

    Metric depth map: depth_m = s / d_rel   (per pixel)
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Optional

import numpy as np
import torch

from .camera import CameraModel

logger = logging.getLogger(__name__)

_MIDAS_REPO = "intel-isl/MiDaS"
_MIDAS_MODEL = "MiDaS_small"
_MIDAS_TRANSFORMS_MODEL = "small"


class DepthEstimator:
    """Monocular depth estimator wrapping Intel MiDaS.

    Set ``enabled=False`` (or ``depth.enabled: false`` in YAML) to disable
    the ~100 MB download and fall back to homography-only geometry.
    """

    def __init__(
        self,
        camera: Optional[CameraModel] = None,
        enabled: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.camera = camera
        self.enabled = enabled
        self._model = None
        self._transform = None

        if not enabled:
            logger.info("DepthEstimator: disabled; homography-only mode active.")
            return

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            logger.info(f"Loading MiDaS ({_MIDAS_MODEL}) via torch.hub …")
            self._model = torch.hub.load(
                _MIDAS_REPO, _MIDAS_MODEL, trust_repo=True
            )
            self._model.eval().to(self._device)

            midas_transforms = torch.hub.load(_MIDAS_REPO, "transforms", trust_repo=True)
            self._transform = getattr(midas_transforms, f"{_MIDAS_TRANSFORMS_MODEL}_transform")
            logger.info("MiDaS loaded successfully.")
        except Exception as exc:
            warnings.warn(
                f"Could not load MiDaS ({exc}). "
                "Depth estimation is disabled for this session. "
                "Install requirements or set depth.enabled=false in config."
            )
            self.enabled = False

    # ------------------------------------------------------------------

    def estimate(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Estimate a relative depth map from an RGB image.

        Args:
            image_rgb: ``(H, W, 3)`` uint8 RGB image.

        Returns:
            ``(H, W)`` float32 relative depth map (larger = closer), or None
            if depth estimation is disabled.
        """
        if not self.enabled or self._model is None:
            return None

        inp = self._transform(image_rgb).to(self._device)
        with torch.no_grad():
            pred = self._model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return pred.cpu().numpy().astype(np.float32)

    def to_metric_depth(self, rel_depth: np.ndarray) -> np.ndarray:
        """Convert MiDaS relative depth to approximate metric depth (metres).

        Requires a ``CameraModel`` for calibration.  Falls back to returning
        normalised relative depth if no camera is set.
        """
        if self.camera is None:
            warnings.warn("No CameraModel set; returning normalised relative depth.")
            d = rel_depth - rel_depth.min()
            rng = d.max()
            return d / rng if rng > 0 else d

        cam = self.camera
        theta = cam.tilt_rad
        if abs(math.sin(theta)) < 1e-6:
            warnings.warn("Camera tilt is ~0°; metric calibration unreliable.")
            return rel_depth

        # Expected metric depth at principal point (ground plane)
        d_ground = cam.height_m / math.sin(theta)

        cy, cx = int(cam.cy), int(cam.cx)
        cy = max(0, min(cy, rel_depth.shape[0] - 1))
        cx = max(0, min(cx, rel_depth.shape[1] - 1))

        d_rel_center = float(rel_depth[cy, cx])
        if d_rel_center < 1e-8:
            warnings.warn("Relative depth at principal point is ~0; check MiDaS output.")
            return rel_depth

        scale = d_ground / d_rel_center
        metric = scale / np.clip(rel_depth, 1e-6, None)
        return metric.astype(np.float32)

    def overlay_depth_on_image(
        self,
        image_rgb: np.ndarray,
        depth_map: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create a colour-mapped depth overlay on *image_rgb*.

        Args:
            image_rgb: ``(H, W, 3)`` uint8 RGB image.
            depth_map: ``(H, W)`` float32 depth map.
            alpha:     Blend factor (0 = image only, 1 = depth only).

        Returns:
            ``(H, W, 3)`` uint8 blended RGB image.
        """
        import cv2

        # Normalise to [0, 255]
        d = depth_map.copy()
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            d = np.zeros_like(d, dtype=np.uint8)

        coloured = cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)
        coloured_rgb = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

        blended = (alpha * coloured_rgb + (1 - alpha) * image_rgb).clip(0, 255).astype(np.uint8)
        return blended


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_depth_estimator(cfg: dict, camera: Optional[CameraModel] = None) -> DepthEstimator:
    """Construct a DepthEstimator from a config dict."""
    return DepthEstimator(
        camera=camera,
        enabled=cfg.get("enabled", False),
        device=cfg.get("device"),
    )
