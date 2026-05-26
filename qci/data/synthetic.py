"""Synthetic crowd dataset — generated on-the-fly with known ground-truth counts.

Each sample is a (C, H, W) float32 image tensor in [0,1], an integer count, and a
(1, H, W) float32 density map that integrates to ``count``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _make_gaussian_kernel_2d(sigma: float, radius: int) -> np.ndarray:
    """2-D Gaussian kernel of shape (2r+1, 2r+1)."""
    size = 2 * radius + 1
    ax = np.arange(size, dtype=np.float64) - radius
    gauss = np.exp(-(ax**2) / (2.0 * sigma**2))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


def make_density_map(
    annotations: np.ndarray,
    image_h: int,
    image_w: int,
    sigma: float = 8.0,
    radius: Optional[int] = None,
    renormalize_border: bool = True,
) -> np.ndarray:
    """Produce a density map from (N, 2) head-annotation array (x, y order).

    Args:
        renormalize_border: When True (default) each truncated border kernel is
            renormalised so the density map integrates exactly to ``N``.
            Set False for ShanghaiTech compatibility (crowds may extend off-frame).

    The density map integrates to ``N`` (the annotation count) when
    ``renormalize_border=True``.
    """
    if radius is None:
        radius = max(1, int(3 * sigma))
    kernel = _make_gaussian_kernel_2d(sigma, radius)
    density = np.zeros((image_h, image_w), dtype=np.float32)

    for x, y in annotations:
        ix, iy = int(round(x)), int(round(y))
        ix = np.clip(ix, 0, image_w - 1)
        iy = np.clip(iy, 0, image_h - 1)

        r0 = iy - radius
        r1 = iy + radius + 1
        c0 = ix - radius
        c1 = ix + radius + 1

        kr0 = max(0, -r0)
        kr1 = kernel.shape[0] - max(0, r1 - image_h)
        kc0 = max(0, -c0)
        kc1 = kernel.shape[1] - max(0, c1 - image_w)

        r0, r1 = max(0, r0), min(image_h, r1)
        c0, c1 = max(0, c0), min(image_w, c1)

        patch = kernel[kr0:kr1, kc0:kc1]
        if renormalize_border:
            patch_sum = patch.sum()
            if patch_sum > 0:
                patch = patch / patch_sum   # each head contributes exactly 1
        density[r0:r1, c0:c1] += patch

    return density


class SyntheticCrowdDataset(Dataset):
    """Tiny synthetic dataset for pipeline testing without external downloads.

    Each image is a random noisy background with *n* coloured ellipses
    representing "heads".  Ground-truth count is exact.
    """

    def __init__(
        self,
        n_images: int = 20,
        image_size: Tuple[int, int] = (256, 256),
        max_count: int = 50,
        density_sigma: float = 8.0,
        seed: int = 42,
    ) -> None:
        self.n_images = n_images
        self.image_size = image_size  # (H, W)
        self.max_count = max_count
        self.density_sigma = density_sigma
        self._rng = np.random.default_rng(seed)
        self._samples = self._generate()

    # ------------------------------------------------------------------

    def _generate(self):
        H, W = self.image_size
        samples = []
        for _ in range(self.n_images):
            count = int(self._rng.integers(1, self.max_count + 1))

            # Background: smooth gradient + noise
            bg = self._rng.random((H, W, 3), dtype=np.float32) * 0.3
            yy, xx = np.mgrid[:H, :W]
            for c in range(3):
                bg[:, :, c] += (
                    0.3 * xx / W * self._rng.random()
                    + 0.3 * yy / H * self._rng.random()
                )
            bg = np.clip(bg, 0.0, 1.0)

            # Annotations: random (x, y) positions
            xs = self._rng.uniform(5, W - 5, size=count)
            ys = self._rng.uniform(5, H - 5, size=count)
            annotations = np.column_stack([xs, ys])

            # Draw ellipses
            img = bg.copy()
            color_palette = self._rng.random((count, 3), dtype=np.float32) * 0.6 + 0.2
            for (x, y), color in zip(annotations, color_palette):
                rx, ry = int(self._rng.integers(3, 8)), int(self._rng.integers(3, 8))
                for dy in range(-ry, ry + 1):
                    for dx in range(-rx, rx + 1):
                        if (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0:
                            py, px = int(y) + dy, int(x) + dx
                            if 0 <= py < H and 0 <= px < W:
                                img[py, px] = color

            density = make_density_map(annotations, H, W, sigma=self.density_sigma)
            samples.append((img, count, density))
        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int):
        img_np, count, density_np = self._samples[idx]
        image = torch.from_numpy(img_np).permute(2, 0, 1).float()   # (3, H, W)
        density = torch.from_numpy(density_np).unsqueeze(0).float()  # (1, H, W)
        return image, count, density
