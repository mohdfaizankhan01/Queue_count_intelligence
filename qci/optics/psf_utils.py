"""PSF creation, manipulation, and convolution utilities."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PSF factories — all return (1, 1, H, W) float32 tensors, energy-normalised
# ---------------------------------------------------------------------------


def _identity_kernel(kernel_size: int) -> torch.Tensor:
    k = torch.zeros(1, 1, kernel_size, kernel_size)
    k[0, 0, kernel_size // 2, kernel_size // 2] = 1.0
    return k


def gaussian_psf(kernel_size: int, sigma: float) -> torch.Tensor:
    """Normalised Gaussian PSF.  Returns (1,1,H,W)."""
    if sigma < 1e-8:
        return _identity_kernel(kernel_size)
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    g = torch.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def disk_psf(kernel_size: int, radius: float) -> torch.Tensor:
    """Normalised disk (pillbox) PSF.  Returns (1,1,H,W)."""
    if radius < 0.5:
        return _identity_kernel(kernel_size)
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    mask = (x**2 + y**2 <= radius**2).float()
    s = mask.sum()
    mask = mask / s if s > 0 else mask
    return mask.unsqueeze(0).unsqueeze(0)


def motion_blur_psf(kernel_size: int, angle_deg: float = 0.0) -> torch.Tensor:
    """Linear motion-blur PSF at *angle_deg*.  Returns (1,1,H,W)."""
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    for i in range(-center, center + 1):
        x = int(round(center + i * cos_a))
        y = int(round(center + i * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            k[y, x] = 1.0
    s = k.sum()
    k = k / s if s > 0 else k
    k[center, center] = 1.0 if s == 0 else k[center, center]
    return torch.from_numpy(k).unsqueeze(0).unsqueeze(0)


def coded_aperture_psf(kernel_size: int, seed: int = 42) -> torch.Tensor:
    """Binary coded-aperture PSF (random, seeded).  Returns (1,1,H,W)."""
    rng = np.random.default_rng(seed)
    pattern = rng.choice([0.0, 1.0], size=(kernel_size, kernel_size)).astype(np.float32)
    s = pattern.sum()
    if s == 0:
        pattern[kernel_size // 2, kernel_size // 2] = 1.0
        s = 1.0
    pattern /= s
    return torch.from_numpy(pattern).unsqueeze(0).unsqueeze(0)


def load_psf_file(path: str | Path, kernel_size: int) -> torch.Tensor:
    """Load PSF from .npy or image file, resize to *kernel_size*.  Returns (1,1,H,W)."""
    import cv2

    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(str(p)).astype(np.float32)
    else:
        arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    arr = cv2.resize(arr, (kernel_size, kernel_size))
    s = arr.sum()
    arr = arr / s if s > 0 else arr
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def energy_normalize(psf: torch.Tensor) -> torch.Tensor:
    """Normalise *psf* so its elements sum to 1."""
    s = psf.sum()
    return psf / s if s.abs() > 1e-10 else psf


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------


def convolve_image(x: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Apply *psf* (1,1,kH,kW) depthwise to image batch *x* (B,C,H,W).

    Preserves spatial dimensions via symmetric padding.
    """
    B, C, H, W = x.shape
    kH, kW = psf.shape[-2], psf.shape[-1]
    pad_h, pad_w = kH // 2, kW // 2

    # Treat every (batch × channel) slice as a 1-channel image
    x_flat = x.reshape(B * C, 1, H, W)
    psf_dev = psf.to(dtype=x.dtype, device=x.device)
    y_flat = F.conv2d(x_flat, psf_dev, padding=(pad_h, pad_w))
    return y_flat.reshape(B, C, H, W)
