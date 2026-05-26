"""Ground-plane homography computation and bird's-eye-view warping.

The ground plane is the world plane Y = 0 (flat floor at camera height = 0).
All coordinates are in metres unless stated otherwise.

Analytical derivation
---------------------
For a pinhole camera with intrinsic matrix K, rotation R (world→cam), and
translation t (world→cam), the projection of a ground-plane point P = (X, 0, Z)
into the image is:

    s·[u, v, 1]ᵀ = K · [R | t] · [X, 0, Z, 1]ᵀ
                 = K · (r₁·X + r₃·Z + t)

where r₁, r₃ are columns 0 and 2 of R.  This defines a 3×3 homography:

    H_gnd→img = K · [r₁ | r₃ | t]          maps (X, Z, 1) → (u·w, v·w, w)
    H_img→gnd = H_gnd→img⁻¹                 maps (u, v, 1) → (X·w, Z·w, w)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .camera import CameraModel


class GroundPlaneMapper:
    """Compute and apply the image ↔ ground-plane homography.

    Parameters
    ----------
    camera:
        ``CameraModel`` instance describing the camera.
    bev_pixels_per_metre:
        Resolution of the bird's-eye-view (BEV) output image.
    bev_coverage_m:
        ``(width_m, depth_m)`` extent of the ground area to render, centred
        horizontally on the camera nadir and starting at the camera base.
    """

    def __init__(
        self,
        camera: CameraModel,
        bev_pixels_per_metre: float = 30.0,
        bev_coverage_m: Tuple[float, float] = (20.0, 30.0),
    ) -> None:
        self.camera = camera
        self.ppm = bev_pixels_per_metre
        self.bev_w_m, self.bev_d_m = bev_coverage_m

        self.H_img_to_gnd: np.ndarray = self._compute_H_img_to_gnd()
        self.H_gnd_to_img: np.ndarray = np.linalg.inv(self.H_img_to_gnd)

        # BEV image dimensions
        self.bev_w = int(self.bev_w_m * self.ppm)
        self.bev_h = int(self.bev_d_m * self.ppm)
        # BEV origin: nadir of camera maps to (bev_w/2, 0)
        self.bev_ox = self.bev_w / 2.0   # pixels
        self.bev_oy = 0.0

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_H_img_to_gnd(self) -> np.ndarray:
        """Compute H_img→gnd analytically from the camera model."""
        cam = self.camera
        R = cam.R_world_to_cam          # (3,3) world→cam
        t = cam.t_world_to_cam          # (3,) world→cam

        r1 = R[:, 0]   # world X axis in camera coords
        r3 = R[:, 2]   # world Z axis in camera coords

        # Homography from ground plane to image
        M = np.column_stack([r1, r3, t])          # (3,3)
        H_gnd_to_img = cam.K @ M                  # (3,3)

        return np.linalg.inv(H_gnd_to_img)

    # ------------------------------------------------------------------
    # Point transforms
    # ------------------------------------------------------------------

    def image_point_to_ground_metres(
        self, pt: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Map image pixel (u, v) to ground-plane (x_m, z_m) in metres."""
        uv1 = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        xzw = self.H_img_to_gnd @ uv1
        return float(xzw[0] / xzw[2]), float(xzw[1] / xzw[2])

    def ground_metres_to_image_point(
        self, x_m: float, z_m: float
    ) -> Tuple[float, float]:
        """Map ground-plane (x_m, z_m) to image pixel (u, v)."""
        xz1 = np.array([x_m, z_m, 1.0], dtype=np.float64)
        uvw = self.H_gnd_to_img @ xz1
        return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])

    def ground_metres_to_bev_pixel(
        self, x_m: float, z_m: float
    ) -> Tuple[float, float]:
        """Map ground (x_m, z_m) to BEV image pixel."""
        u_bev = x_m * self.ppm + self.bev_ox
        v_bev = z_m * self.ppm + self.bev_oy
        return u_bev, v_bev

    # ------------------------------------------------------------------
    # Image warping
    # ------------------------------------------------------------------

    def warp_to_birdseye(self, image: np.ndarray) -> np.ndarray:
        """Warp a camera image to a top-down bird's-eye view.

        Args:
            image: ``(H, W, C)`` uint8 BGR/RGB image from the camera.

        Returns:
            ``(bev_h, bev_w, C)`` uint8 BEV image.
        """
        # H that maps a BEV pixel (u_bev, v_bev) → ground (x_m, z_m):
        #   x_m = (u_bev - bev_ox) / ppm
        #   z_m = (v_bev - bev_oy) / ppm
        # Then ground → image pixel via H_gnd_to_img.
        # Compose into a single H: BEV pixel → image pixel.
        ppm = self.ppm
        H_bev_to_gnd = np.array(
            [[1/ppm, 0,     -self.bev_ox / ppm],
             [0,     1/ppm, -self.bev_oy / ppm],
             [0,     0,      1.0]],
            dtype=np.float64,
        )
        H_bev_to_img = self.H_gnd_to_img @ H_bev_to_gnd

        # cv2.warpPerspective(src, M, dsize): M maps src→dst, or equivalently
        # dst pixel obtains its value from M⁻¹(dst_pixel) in src.
        # We want dst=BEV ← src=image, so M should map image→BEV:
        H_img_to_bev = np.linalg.inv(H_bev_to_img)

        bev = cv2.warpPerspective(
            image,
            H_img_to_bev,
            (self.bev_w, self.bev_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return bev

    def project_density_to_ground(
        self, density_map: np.ndarray
    ) -> np.ndarray:
        """Warp a density map from camera perspective to BEV ground plane.

        Args:
            density_map: ``(H, W)`` float32 density map.

        Returns:
            ``(bev_h, bev_w)`` float32 BEV density map.
        """
        ppm = self.ppm
        H_bev_to_gnd = np.array(
            [[1/ppm, 0,     -self.bev_ox / ppm],
             [0,     1/ppm, -self.bev_oy / ppm],
             [0,     0,      1.0]],
            dtype=np.float64,
        )
        H_bev_to_img = self.H_gnd_to_img @ H_bev_to_gnd
        H_img_to_bev = np.linalg.inv(H_bev_to_img)

        bev_dmap = cv2.warpPerspective(
            density_map.astype(np.float32),
            H_img_to_bev,
            (self.bev_w, self.bev_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return bev_dmap


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def compute_homography_from_camera_model(camera: CameraModel) -> np.ndarray:
    """Return H_img→gnd (3×3) for *camera*. det ≠ 0 by construction."""
    mapper = GroundPlaneMapper(camera)
    return mapper.H_img_to_gnd
