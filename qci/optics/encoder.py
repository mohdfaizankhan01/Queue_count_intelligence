"""OpticalEncoder: differentiable optical front-end x -> y."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .psf_utils import (
    coded_aperture_psf,
    convolve_image,
    disk_psf,
    gaussian_psf,
    load_psf_file,
)


class OpticalEncoder(nn.Module):
    """Differentiable optical encoder mapping clean image x -> measurement y.

    Three encoding modes are supported:

    * **defocus** — Gaussian (or disk) PSF whose blur radius is proportional to
      ``strength``.  No trainable parameters.
    * **coded_mask** — fixed binary coded-aperture PSF loaded from file or
      auto-generated.  ``strength`` interpolates between identity and the mask.
    * **learnable_psf** — PSF stored as an ``nn.Parameter``; gradients flow
      through it for end-to-end optimisation.  ``strength`` still controls the
      identity–PSF interpolation so the sweep remains meaningful.

    In all modes ``strength ∈ [0, 1]`` acts as the severity knob:
    ``strength=0`` → identity passthrough; ``strength=1`` → maximum encoding.
    """

    def __init__(
        self,
        mode: Literal["defocus", "coded_mask", "learnable_psf"] = "defocus",
        strength: float = 0.5,
        kernel_size: int = 31,
        sigma_max: float = 10.0,
        radius_max: float = 15.0,
        psf_shape: Literal["gaussian", "disk"] = "gaussian",
        coded_mask_path: Optional[str] = None,
        coded_mask_seed: int = 42,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        if mode not in ("defocus", "coded_mask", "learnable_psf"):
            raise ValueError(f"Unknown mode: {mode!r}")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.mode = mode
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.kernel_size = kernel_size
        self.sigma_max = sigma_max
        self.radius_max = radius_max
        self.psf_shape = psf_shape
        self.in_channels = in_channels

        if mode == "coded_mask":
            if coded_mask_path is not None:
                psf = load_psf_file(coded_mask_path, kernel_size)
            else:
                psf = coded_aperture_psf(kernel_size, seed=coded_mask_seed)
            self.register_buffer("_coded_psf", psf)  # (1,1,kH,kW)

        elif mode == "learnable_psf":
            # Initialise from a soft Gaussian so early-training outputs are sensible
            init = gaussian_psf(kernel_size, sigma=2.0)
            # Store raw (unnormalised) logits; softmax applied in _get_psf
            self._learnable_psf = nn.Parameter(
                torch.log(init.clamp(min=1e-8))
            )

    # ------------------------------------------------------------------
    # PSF helpers
    # ------------------------------------------------------------------

    def _identity_kernel(self) -> torch.Tensor:
        k = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        k[0, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        return k

    def _get_psf(self) -> torch.Tensor:
        """Return the effective PSF as a (1,1,kH,kW) tensor (on CPU)."""
        ks = self.kernel_size

        if self.mode == "defocus":
            if self.psf_shape == "disk":
                radius = self.strength * self.radius_max
                psf_full = disk_psf(ks, radius)
            else:
                sigma = self.strength * self.sigma_max
                psf_full = gaussian_psf(ks, sigma)
            # Interpolate with identity for smooth sweep (handled implicitly:
            # gaussian_psf/disk_psf already return identity when sigma/radius≈0)
            return psf_full

        elif self.mode == "coded_mask":
            identity = self._identity_kernel()
            # Blend: strength=0 → identity, strength=1 → coded mask
            psf = (1.0 - self.strength) * identity + self.strength * self._coded_psf.cpu()
            # Renormalise after linear mix
            psf = psf / (psf.sum() + 1e-10)
            return psf

        else:  # learnable_psf
            # Softmax normalisation ensures PSF sums to 1 (energy conservation)
            logits_flat = self._learnable_psf.view(1, 1, -1)
            psf_norm = F.softmax(logits_flat, dim=-1).view(self._learnable_psf.shape)
            # Blend with identity
            identity = torch.zeros_like(psf_norm)
            identity[0, 0, ks // 2, ks // 2] = 1.0
            psf = (1.0 - self.strength) * identity + self.strength * psf_norm
            return psf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_strength(self, strength: float) -> None:
        """Update encoding strength for sweep evaluation (non-differentiable)."""
        self.strength = float(np.clip(strength, 0.0, 1.0))

    def get_psf_tensor(self) -> torch.Tensor:
        """Return current effective PSF as (1,1,kH,kW) for inspection / Wiener filter."""
        with torch.no_grad():
            return self._get_psf().detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image batch.

        Args:
            x: ``(B, C, H, W)`` image tensor with values in ``[0, 1]``.

        Returns:
            ``(B, C, H, W)`` encoded measurement, clamped to ``[0, 1]``.
        """
        if self.strength < 1e-7:
            return x

        psf = self._get_psf().to(device=x.device, dtype=x.dtype)
        y = convolve_image(x, psf)
        return torch.clamp(y, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_encoder(cfg: dict) -> OpticalEncoder:
    """Construct an OpticalEncoder from a config dict."""
    return OpticalEncoder(
        mode=cfg.get("mode", "defocus"),
        strength=cfg.get("strength", 0.5),
        kernel_size=cfg.get("kernel_size", 31),
        sigma_max=cfg.get("sigma_max", 10.0),
        radius_max=cfg.get("radius_max", 15.0),
        psf_shape=cfg.get("psf_shape", "gaussian"),
        coded_mask_path=cfg.get("coded_mask_path"),
        coded_mask_seed=cfg.get("coded_mask_seed", 42),
    )
