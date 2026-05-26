#!/usr/bin/env python3
"""Geometry pipeline: bird's-eye-view projection, optional depth, queue length.

Usage::

    # With a real image
    python scripts/run_geometry.py --image path/to/frame.jpg

    # Synthetic test (no real image needed)
    python scripts/run_geometry.py --synthetic

    # With custom camera config
    python scripts/run_geometry.py --image frame.jpg --camera configs/camera.yaml

    # With MiDaS depth (downloads ~100 MB on first run)
    python scripts/run_geometry.py --synthetic --depth

Outputs (written to ``results/geometry/``)::

    birdseye.png         — top-down BEV projection
    depth_overlay.png    — metric-depth colour overlay (if --depth)
    queue_geometry.json  — QueueGeometry fields as JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qci.geometry.camera import CameraModel
from qci.geometry.depth import build_depth_estimator
from qci.geometry.homography import GroundPlaneMapper
from qci.geometry.queue_length import QueueLengthEstimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_geometry")


# ---------------------------------------------------------------------------
# Synthetic test-image generator
# ---------------------------------------------------------------------------


def _make_synthetic_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Generate a simple synthetic scene (gradient background + white blobs)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky-ish gradient
    for row in range(h):
        val = int(50 + 150 * row / h)
        img[row, :] = [val // 2, val // 2, val]
    # Fake "people" blobs
    rng = np.random.default_rng(42)
    for _ in range(15):
        cx = int(rng.integers(40, w - 40))
        cy = int(rng.integers(h // 3, h - 30))
        cv2.ellipse(img, (cx, cy), (10, 20), 0, 0, 360, (200, 180, 160), -1)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="QCI geometry pipeline")
    parser.add_argument("--image", help="Path to input image (JPG/PNG)")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use a generated synthetic image"
    )
    parser.add_argument(
        "--camera",
        help="Path to YAML file with a 'camera' section. "
        "Defaults to typical_polling_station preset.",
    )
    parser.add_argument("--depth", action="store_true", help="Enable MiDaS depth estimation")
    parser.add_argument("--out-dir", default="results/geometry", help="Output directory")
    args = parser.parse_args()

    if not args.image and not args.synthetic:
        parser.error("Provide --image PATH or --synthetic")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load camera model ---
    if args.camera:
        camera = CameraModel.from_yaml(args.camera)
    else:
        camera = CameraModel.typical_polling_station()
    logger.info(f"Camera: h={camera.height_m}m, tilt={camera.tilt_deg}°, f={camera.focal_length_px}px")

    # --- Load / generate image ---
    if args.synthetic:
        image_bgr = cv2.cvtColor(
            _make_synthetic_image(camera.image_height, camera.image_width),
            cv2.COLOR_RGB2BGR,
        )
        logger.info("Using synthetic image")
    else:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot open image: {args.image}")
        # Resize to camera resolution if different
        if image_bgr.shape[:2] != (camera.image_height, camera.image_width):
            image_bgr = cv2.resize(image_bgr, (camera.image_width, camera.image_height))

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --- Ground-plane projection ---
    mapper = GroundPlaneMapper(
        camera, bev_pixels_per_metre=30.0, bev_coverage_m=(20.0, 30.0)
    )
    bev = mapper.warp_to_birdseye(image_bgr)
    bev_path = out_dir / "birdseye.png"
    cv2.imwrite(str(bev_path), bev)
    logger.info(f"BEV image saved to {bev_path}")

    # --- Depth estimation (optional) ---
    depth_map = None
    if args.depth:
        de = build_depth_estimator({"enabled": True}, camera=camera)
        rel_depth = de.estimate(image_rgb)
        if rel_depth is not None:
            depth_map = de.to_metric_depth(rel_depth)
            overlay = de.overlay_depth_on_image(image_rgb, depth_map)
            depth_path = out_dir / "depth_overlay.png"
            cv2.imwrite(str(depth_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            logger.info(f"Depth overlay saved to {depth_path}")
    else:
        logger.info("Depth disabled (pass --depth to enable MiDaS)")

    # --- Queue-length estimation from BEV density proxy ---
    # Use BEV brightness as a stand-in for density when no counter output is available
    bev_gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mpp = 1.0 / mapper.ppm  # metres per BEV pixel
    estimator = QueueLengthEstimator(min_persons=2)
    geo = estimator.from_density_map(
        bev_gray,
        metres_per_pixel=mpp,
        origin_x_m=-mapper.bev_w_m / 2.0,
        origin_z_m=0.0,
    )

    geo_dict = asdict(geo)
    geo_path = out_dir / "queue_geometry.json"
    with open(geo_path, "w") as f:
        json.dump(geo_dict, f, indent=2)
    logger.info(f"Queue geometry saved to {geo_path}")

    print("\n=== Queue Geometry ===")
    for k, v in geo_dict.items():
        print(f"  {k:25s}: {v:.3f}" if isinstance(v, float) else f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()
