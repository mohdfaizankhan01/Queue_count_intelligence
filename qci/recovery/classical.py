"""Classical and learned restoration implementations.

* ``WienerRestoration`` — non-blind Wiener deconvolution (requires known PSF).
* ``UNetStub`` — tiny U-Net stub; randomly initialised, ready for fine-tuning.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RestorationModule


# ---------------------------------------------------------------------------
# Wiener deconvolution
# ---------------------------------------------------------------------------


class WienerRestoration(RestorationModule):
    """Non-blind Wiener deconvolution using the known encoder PSF.

    The PSF can be supplied at construction time or updated later via
    ``set_psf()``.  Without a PSF it falls back to identity.

    Wiener filter (freq. domain):
        W = H* / (|H|^2 + K)
        X̂  = W · Y
    """

    def __init__(
        self,
        psf: Optional[torch.Tensor] = None,
        nsr: float = 0.01,
    ) -> None:
        super().__init__()
        self.nsr = nsr
        if psf is not None:
            self.register_buffer("_psf", psf.float())
        else:
            self._psf = None

    def set_psf(self, psf: torch.Tensor) -> None:
        """Update the PSF (e.g. after each encoding-strength change)."""
        self._psf = psf.float().to(next(self.parameters(), torch.tensor(0.0)).device
                                   if len(list(self.parameters())) else "cpu")

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if self._psf is None:
            return y

        B, C, H, W = y.shape
        psf = self._psf.to(device=y.device, dtype=y.dtype)
        kH, kW = psf.shape[-2], psf.shape[-1]

        # Pad PSF to image size and shift to DC-at-corner convention
        psf_padded = torch.zeros(1, 1, H, W, device=y.device, dtype=y.dtype)
        cy = H // 2 - kH // 2
        cx = W // 2 - kW // 2
        psf_padded[0, 0, cy:cy + kH, cx:cx + kW] = psf[0, 0]
        psf_shifted = torch.fft.ifftshift(psf_padded, dim=(-2, -1))

        # Process each channel independently (same PSF for all)
        results = []
        for c in range(C):
            Y = torch.fft.rfft2(y[:, c:c + 1, :, :])                  # (B,1,H,W/2+1)
            H_fft = torch.fft.rfft2(psf_shifted.expand(B, 1, H, W))   # (B,1,H,W/2+1)
            W_fft = torch.conj(H_fft) / (H_fft.abs() ** 2 + self.nsr)
            x_hat = torch.fft.irfft2(W_fft * Y, s=(H, W))
            results.append(x_hat)

        return torch.cat(results, dim=1).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Tiny U-Net stub
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetStub(RestorationModule):
    """Tiny 2-level U-Net for learned restoration.

    Randomly initialised — intended as a stub for future fine-tuning.
    The architecture is kept small (≈280 K parameters) so it can run on CPU
    during testing without significant overhead.
    """

    def __init__(self, in_channels: int = 3, features: int = 32) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock(features * 2, features * 4)

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec2 = _ConvBlock(features * 4, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec1 = _ConvBlock(features * 2, features)

        self.out_conv = nn.Conv2d(features, in_channels, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # noqa: D102
        e1 = self.enc1(y)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))
