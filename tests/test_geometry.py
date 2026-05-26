"""Layer 4 geometry tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qci.geometry.camera import CameraModel
from qci.geometry.homography import (
    GroundPlaneMapper,
    compute_homography_from_camera_model,
)
from qci.geometry.queue_length import QueueGeometry, QueueLengthEstimator


# ---------------------------------------------------------------------------
# CameraModel
# ---------------------------------------------------------------------------


class TestCameraModel:
    def test_preset_fields(self):
        cam = CameraModel.typical_polling_station()
        assert cam.height_m == 3.0
        assert cam.tilt_deg == 30.0
        assert cam.focal_length_px == 1200.0
        assert cam.image_width == 1920
        assert cam.image_height == 1080

    def test_principal_point(self):
        cam = CameraModel(image_width=1920, image_height=1080)
        assert cam.cx == 960.0
        assert cam.cy == 540.0

    def test_K_shape(self):
        cam = CameraModel.typical_polling_station()
        assert cam.K.shape == (3, 3)
        assert cam.K[0, 0] == cam.focal_length_px

    def test_R_orthogonal(self):
        cam = CameraModel.typical_polling_station()
        R = cam.R_world_to_cam
        diff = np.abs(R @ R.T - np.eye(3))
        assert diff.max() < 1e-10, "R is not orthogonal"

    def test_R_det_one(self):
        cam = CameraModel.typical_polling_station()
        assert abs(np.linalg.det(cam.R_world_to_cam) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------


class TestHomography:
    @pytest.fixture
    def cam(self) -> CameraModel:
        return CameraModel.typical_polling_station()

    def test_H_shape(self, cam):
        H = compute_homography_from_camera_model(cam)
        assert H.shape == (3, 3)

    def test_H_det_nonzero(self, cam):
        """H must be invertible.  We check the forward map H_gnd→img because
        its det ~4 M (focal_px² × scene_scale) while H_img→gnd is its inverse
        and has a proportionally tiny det; testing absolute value of a det is
        scale-dependent, so we test the condition number instead."""
        H = compute_homography_from_camera_model(cam)
        # Round-trip: H_img→gnd then its inverse must recover identity
        H_inv = np.linalg.inv(H)
        residual = np.abs(H_inv @ H - np.eye(3)).max()
        assert residual < 1e-8, f"H is numerically singular (residual={residual:.2e})"
        # Condition number of H_img→gnd should be finite and reasonable
        assert np.linalg.cond(H) < 1e10, "Homography is ill-conditioned"

    def test_round_trip_within_1cm(self, cam):
        """image_point → ground_metres → image_point should be < 1px (≈1cm)."""
        mapper = GroundPlaneMapper(cam)
        # Pick a ground point in front of the camera at known depth
        x_m, z_m = 2.5, 8.0   # 2.5 m right, 8 m in front
        u_img, v_img = mapper.ground_metres_to_image_point(x_m, z_m)

        # Round-trip: image → ground
        x_rt, z_rt = mapper.image_point_to_ground_metres((u_img, v_img))
        err_x = abs(x_rt - x_m)
        err_z = abs(z_rt - z_m)
        assert err_x < 0.01, f"X round-trip error {err_x:.4f} m ≥ 1 cm"
        assert err_z < 0.01, f"Z round-trip error {err_z:.4f} m ≥ 1 cm"

    def test_nadir_maps_to_image_centre_x(self, cam):
        """Ground point directly below camera (X=0) should map to u=cx."""
        mapper = GroundPlaneMapper(cam)
        u, _ = mapper.ground_metres_to_image_point(0.0, 5.0)
        assert abs(u - cam.cx) < 1.0, f"Nadir X not at cx: u={u:.1f}, cx={cam.cx}"

    def test_bev_warp_shape(self, cam):
        import cv2
        import numpy as np
        mapper = GroundPlaneMapper(cam, bev_pixels_per_metre=20.0, bev_coverage_m=(10.0, 15.0))
        img = np.zeros((cam.image_height, cam.image_width, 3), dtype=np.uint8)
        bev = mapper.warp_to_birdseye(img)
        expected_w = int(10.0 * 20.0)
        expected_h = int(15.0 * 20.0)
        assert bev.shape == (expected_h, expected_w, 3)


# ---------------------------------------------------------------------------
# QueueLengthEstimator
# ---------------------------------------------------------------------------


class TestQueueLengthEstimator:
    def test_empty_points(self):
        est = QueueLengthEstimator()
        geo = est.from_points([])
        assert geo.length_m == 0.0
        assert geo.person_count == 0.0

    def test_single_point(self):
        est = QueueLengthEstimator(min_persons=2)
        geo = est.from_points([(1.0, 2.0)])
        assert geo.length_m == 0.0   # fewer than min_persons

    def test_length_positive_for_two_points(self):
        est = QueueLengthEstimator(min_persons=2)
        geo = est.from_points([(0.0, 0.0), (10.0, 0.0)])
        assert geo.length_m > 0.0

    def test_elongated_cloud_along_z(self):
        """A cloud that is 20 m long along Z and 2 m wide along X.

        PCA should identify the long axis as ≈ PC1 → length_m ≈ 20.
        """
        rng = np.random.default_rng(0)
        n = 200
        points = list(zip(
            rng.normal(0, 1.0, n).tolist(),   # X: narrow
            rng.uniform(0, 20, n).tolist(),    # Z: long
        ))
        est = QueueLengthEstimator()
        geo = est.from_points(points)
        # Length should be the long dimension
        assert geo.length_m > geo.width_m, (
            f"Expected length ({geo.length_m:.1f}) > width ({geo.width_m:.1f})"
        )
        assert geo.length_m > 15.0, f"Length {geo.length_m:.1f} m unexpectedly short"

    def test_from_density_map_non_trivial(self):
        rng = np.random.default_rng(1)
        dmap = rng.random((64, 64)).astype(np.float32)
        est = QueueLengthEstimator()
        geo = est.from_density_map(dmap, metres_per_pixel=0.1)
        assert geo.person_count > 0.0

    def test_geometry_fields_finite(self):
        rng = np.random.default_rng(2)
        pts = list(zip(rng.uniform(-5, 5, 50).tolist(), rng.uniform(0, 20, 50).tolist()))
        est = QueueLengthEstimator()
        geo = est.from_points(pts)
        for field in (geo.length_m, geo.width_m, geo.density_per_sqm,
                      geo.centroid_x_m, geo.centroid_z_m, geo.angle_deg):
            assert math.isfinite(field), f"Non-finite field: {field}"
