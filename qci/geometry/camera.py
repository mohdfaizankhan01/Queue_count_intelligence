"""CameraModel — pinhole camera parameters for a surveillance-mounted camera."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


@dataclass
class CameraModel:
    """Pinhole camera parameters for a fixed, downward-tilted surveillance camera.

    Coordinate convention
    ---------------------
    * World: right-handed, X right, Y up, Z forward.
    * Camera: looking along its +Z axis; +Y is image-up, +X is image-right.
    * The camera sits at height ``height_m`` above the ground plane (Y = 0).
    * ``tilt_deg`` is the downward tilt angle from horizontal (0 = horizontal,
      90 = nadir).

    Attributes:
        height_m:        Camera height above ground in metres.
        tilt_deg:        Downward tilt from horizontal (degrees).
        focal_length_px: Focal length in pixels (assumed equal in x/y).
        image_width:     Frame width in pixels.
        image_height:    Frame height in pixels.
    """

    height_m: float = 3.0
    tilt_deg: float = 30.0
    focal_length_px: float = 1200.0
    image_width: int = 1920
    image_height: int = 1080

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def cx(self) -> float:
        """Principal point x (image centre)."""
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        """Principal point y (image centre)."""
        return self.image_height / 2.0

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        f = self.focal_length_px
        return np.array(
            [[f, 0, self.cx],
             [0, f, self.cy],
             [0, 0, 1.0]],
            dtype=np.float64,
        )

    @property
    def R_world_to_cam(self) -> np.ndarray:
        """3×3 rotation from world to camera coordinates.

        The camera is tilted downward by ``tilt_deg`` around the world X axis.
        """
        theta = math.radians(self.tilt_deg)
        ct, st = math.cos(theta), math.sin(theta)
        return np.array(
            [[1,  0,   0],
             [0,  ct,  st],
             [0, -st,  ct]],
            dtype=np.float64,
        )

    @property
    def t_world_to_cam(self) -> np.ndarray:
        """3-vector translation from world to camera frame.

        Camera centre in world is (0, height_m, 0).
        """
        return -(self.R_world_to_cam @ np.array([0.0, self.height_m, 0.0]))

    @property
    def tilt_rad(self) -> float:
        return math.radians(self.tilt_deg)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, key: str = "camera") -> "CameraModel":
        """Load from a YAML file under *key*."""
        with open(path) as f:
            d = yaml.safe_load(f)
        params = d.get(key, d)
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

    @classmethod
    def typical_polling_station(cls) -> "CameraModel":
        """Default preset for a typical polling-station overhead camera."""
        return cls(
            height_m=3.0,
            tilt_deg=30.0,
            focal_length_px=1200.0,
            image_width=1920,
            image_height=1080,
        )
