"""ShanghaiTech crowd-counting dataset loader (Part A and Part B).

Expected directory layout::

    <root>/
      part_A/
        train_data/
          images/   *.jpg
          ground-truth/  GT_*.mat
        test_data/
          images/
          ground-truth/
      part_B/
        ...

Ground-truth .mat files contain ``image_info[0][0][0][0][0]`` with shape
``(N, 2)`` of (x, y) head locations.

If the dataset is not present the loader raises ``FileNotFoundError`` with a
helpful message pointing to the download URL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import make_density_map

try:
    import scipy.io as sio
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class ShanghaiTechDataset(Dataset):
    """PyTorch Dataset for ShanghaiTech Part A / B crowd counting.

    Returns ``(image, gt_count, density_map)`` where:

    * ``image`` — ``(3, H, W)`` float32 tensor in ``[0, 1]``
    * ``gt_count`` — Python int
    * ``density_map`` — ``(1, H, W)`` float32 tensor integrating to ``gt_count``
    """

    DOWNLOAD_URL = (
        "https://github.com/desenzhou/ShanghaiTechDataset  "
        "(or kaggle: 'tthien/shanghaitech')"
    )

    def __init__(
        self,
        root: str | Path,
        part: Literal["A", "B"] = "A",
        split: Literal["train", "test"] = "test",
        density_sigma: float = 15.0,
        max_images: Optional[int] = None,
    ) -> None:
        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy is required to load ShanghaiTech .mat files")
        if not _CV2_AVAILABLE:
            raise ImportError("opencv-python is required for image loading")

        self.root = Path(root)
        self.part = part
        self.split = split
        self.density_sigma = density_sigma

        data_dir = self.root / f"part_{part}" / f"{split}_data"
        img_dir = data_dir / "images"
        gt_dir = data_dir / "ground-truth"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"ShanghaiTech images not found at {img_dir}.\n"
                f"Download from: {self.DOWNLOAD_URL}"
            )

        self.image_paths = sorted(img_dir.glob("*.jpg"))
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        self.gt_paths = [
            gt_dir / ("GT_" + p.name.replace(".jpg", ".mat"))
            for p in self.image_paths
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        import cv2

        img_bgr = cv2.imread(str(self.image_paths[idx]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = img_rgb.shape[:2]

        mat = sio.loadmat(str(self.gt_paths[idx]))
        # Standard ShanghaiTech mat structure
        annotations = mat["image_info"][0][0][0][0][0].astype(np.float32)  # (N, 2) (x, y)

        gt_count = len(annotations)
        density = make_density_map(annotations, H, W, sigma=self.density_sigma)

        image = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        density_t = torch.from_numpy(density).unsqueeze(0).float()
        return image, gt_count, density_t
