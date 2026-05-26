"""Queue-length estimation from a crowd point cloud or density map.

Algorithm
---------
1. Build a weighted point cloud on the ground plane (x_m, z_m).
2. Run PCA to find the principal axis of the crowd footprint.
3. Queue length  = spread along PC1 (long axis).
4. Queue width   = spread along PC2 (short axis).
5. Density       = person_count / footprint_area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class QueueGeometry:
    """Geometric characterisation of the detected queue."""

    length_m: float          # extent along principal axis
    width_m: float           # extent along minor axis
    density_per_sqm: float   # persons / m²  (0 if area == 0)
    person_count: float      # total crowd count used to produce this estimate
    centroid_x_m: float = 0.0
    centroid_z_m: float = 0.0
    angle_deg: float = 0.0   # orientation of the queue axis (degrees from Z-forward)


class QueueLengthEstimator:
    """Estimate queue length and width from a 2-D ground-plane point cloud.

    Accepts either:
    * A list of (x_m, z_m) person locations, optionally weighted.
    * A 2-D density map on the BEV ground plane with a known scale
      (``metres_per_pixel``).
    """

    def __init__(self, min_persons: int = 2) -> None:
        self.min_persons = min_persons

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def from_points(
        self,
        points: Sequence[Tuple[float, float]],
        weights: Optional[Sequence[float]] = None,
    ) -> QueueGeometry:
        """Estimate queue geometry from ground-plane point locations.

        Args:
            points:  Sequence of (x_m, z_m) locations.
            weights: Optional per-point weights (e.g. detection confidences).
                     Uniform if None.

        Returns:
            ``QueueGeometry`` with length, width, density, and orientation.
        """
        pts = np.array(points, dtype=np.float64)
        if len(pts) == 0:
            return QueueGeometry(0.0, 0.0, 0.0, 0.0)

        w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)

        total_count = float(len(pts))
        centroid = (pts * w[:, None]).sum(axis=0)

        if len(pts) < self.min_persons:
            return QueueGeometry(0.0, 0.0, 0.0, total_count,
                                 float(centroid[0]), float(centroid[1]))

        # Weighted PCA
        centred = pts - centroid
        cov = (centred * w[:, None]).T @ centred   # (2,2)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns ascending order → last is PC1
        pc1 = eigenvectors[:, -1]
        pc2 = eigenvectors[:, 0]

        proj1 = centred @ pc1
        proj2 = centred @ pc2

        length_m = float(proj1.max() - proj1.min())
        width_m = float(proj2.max() - proj2.min())

        area = max(length_m * width_m, 1e-6)
        density = total_count / area
        angle_deg = float(math.degrees(math.atan2(pc1[0], pc1[1])))

        return QueueGeometry(
            length_m=length_m,
            width_m=width_m,
            density_per_sqm=density,
            person_count=total_count,
            centroid_x_m=float(centroid[0]),
            centroid_z_m=float(centroid[1]),
            angle_deg=angle_deg,
        )

    def from_density_map(
        self,
        density_map: np.ndarray,
        metres_per_pixel: float,
        origin_x_m: float = 0.0,
        origin_z_m: float = 0.0,
    ) -> QueueGeometry:
        """Estimate queue geometry from a BEV density map.

        Args:
            density_map:     ``(H, W)`` float32 density map on the BEV plane.
            metres_per_pixel: Physical scale of the BEV image.
            origin_x_m, origin_z_m: World-coords of BEV pixel (0, 0).

        Returns:
            ``QueueGeometry`` derived from the density-weighted centroid cloud.
        """
        H, W = density_map.shape
        if density_map.sum() < 1e-8:
            return QueueGeometry(0.0, 0.0, 0.0, 0.0)

        ys, xs = np.mgrid[:H, :W]
        x_m = xs * metres_per_pixel + origin_x_m
        z_m = ys * metres_per_pixel + origin_z_m

        flat_w = density_map.ravel()
        flat_x = x_m.ravel()
        flat_z = z_m.ravel()

        # Keep pixels with non-negligible density to form the "point cloud"
        threshold = flat_w.max() * 0.01
        mask = flat_w > threshold
        if mask.sum() < self.min_persons:
            total = float(density_map.sum())
            return QueueGeometry(0.0, 0.0, 0.0, total)

        points = list(zip(flat_x[mask].tolist(), flat_z[mask].tolist()))
        weights = flat_w[mask].tolist()
        total_count = float(density_map.sum())

        geo = self.from_points(points, weights)
        # Override count with actual density sum
        return QueueGeometry(
            length_m=geo.length_m,
            width_m=geo.width_m,
            density_per_sqm=total_count / max(geo.length_m * geo.width_m, 1e-6),
            person_count=total_count,
            centroid_x_m=geo.centroid_x_m,
            centroid_z_m=geo.centroid_z_m,
            angle_deg=geo.angle_deg,
        )
