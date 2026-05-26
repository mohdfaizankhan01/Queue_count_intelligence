"""DegradationSim: compose-able realistic sensor degradation pipeline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .psf_utils import convolve_image, motion_blur_psf


class DegradationSim(nn.Module):
    """Simulate realistic sensor and capture degradations.

    Effects are applied in order: motion_blur → low_light → gaussian_noise →
    poisson_noise → downsample.  Each effect has a ``severity ∈ [0, 1]``
    parameter and can be toggled independently.  Severity=0 for any effect
    leaves the image unchanged.

    The module is *not* intended for gradient-based PSF optimisation (Poisson
    sampling is stochastic); it is differentiable w.r.t. the image for all
    effects except Poisson noise.
    """

    def __init__(
        self,
        gaussian_noise: float = 0.0,
        poisson_noise: float = 0.0,
        motion_blur: float = 0.0,
        motion_angle_deg: float = 0.0,
        low_light: float = 0.0,
        downsample: float = 0.0,
        enabled: Optional[dict[str, bool]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.gaussian_noise = float(gaussian_noise)
        self.poisson_noise = float(poisson_noise)
        self.motion_blur = float(motion_blur)
        self.motion_angle_deg = float(motion_angle_deg)
        self.low_light = float(low_light)
        self.downsample = float(downsample)

        _default_enabled = {
            "gaussian_noise": True,
            "poisson_noise": True,
            "motion_blur": True,
            "low_light": True,
            "downsample": True,
        }
        if enabled is not None:
            _default_enabled.update(enabled)
        self.enabled = _default_enabled

        self._rng: Optional[torch.Generator] = None
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)

    # ------------------------------------------------------------------
    # Individual effects
    # ------------------------------------------------------------------

    def _apply_motion_blur(self, x: torch.Tensor, severity: float) -> torch.Tensor:
        ks = max(3, int(severity * 31))
        if ks % 2 == 0:
            ks += 1
        psf = motion_blur_psf(ks, angle_deg=self.motion_angle_deg).to(device=x.device, dtype=x.dtype)
        return convolve_image(x, psf).clamp(0.0, 1.0)

    def _apply_low_light(self, x: torch.Tensor, severity: float) -> torch.Tensor:
        # Darken via gamma compression then reduce exposure
        gamma = 1.0 + severity * 3.0   # 1 → 4
        exposure = 1.0 - severity * 0.7  # 1 → 0.3
        x = x.clamp(1e-8, 1.0).pow(gamma)
        return (x * exposure).clamp(0.0, 1.0)

    def _apply_gaussian_noise(self, x: torch.Tensor, severity: float) -> torch.Tensor:
        std = severity * 0.15
        if self._rng is not None:
            noise = torch.randn(x.shape, generator=self._rng, device=x.device, dtype=x.dtype)
        else:
            noise = torch.randn_like(x)
        return (x + noise * std).clamp(0.0, 1.0)

    def _apply_poisson_noise(self, x: torch.Tensor, severity: float) -> torch.Tensor:
        # Higher severity = fewer photons = noisier
        scale = max(1.0, 255.0 * (1.0 - severity * 0.95))
        photons = torch.poisson(x.clamp(min=0.0) * scale)
        return (photons / scale).clamp(0.0, 1.0)

    def _apply_downsample(self, x: torch.Tensor, severity: float) -> torch.Tensor:
        scale = max(0.1, 1.0 - severity * 0.75)
        B, C, H, W = x.shape
        sh, sw = max(1, int(H * scale)), max(1, int(W * scale))
        small = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enabled degradation effects to image batch *x* (B,C,H,W)."""
        if self.enabled.get("motion_blur", True) and self.motion_blur > 1e-7:
            x = self._apply_motion_blur(x, self.motion_blur)
        if self.enabled.get("low_light", True) and self.low_light > 1e-7:
            x = self._apply_low_light(x, self.low_light)
        if self.enabled.get("gaussian_noise", True) and self.gaussian_noise > 1e-7:
            x = self._apply_gaussian_noise(x, self.gaussian_noise)
        if self.enabled.get("poisson_noise", True) and self.poisson_noise > 1e-7:
            x = self._apply_poisson_noise(x, self.poisson_noise)
        if self.enabled.get("downsample", True) and self.downsample > 1e-7:
            x = self._apply_downsample(x, self.downsample)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_degradation(cfg: dict) -> Optional[DegradationSim]:
    """Return a DegradationSim if ``cfg['enabled']`` is True, else None."""
    if not cfg.get("enabled", False):
        return None
    return DegradationSim(
        gaussian_noise=cfg.get("gaussian_noise", 0.0),
        poisson_noise=cfg.get("poisson_noise", 0.0),
        motion_blur=cfg.get("motion_blur", 0.0),
        motion_angle_deg=cfg.get("motion_angle_deg", 0.0),
        low_light=cfg.get("low_light", 0.0),
        downsample=cfg.get("downsample", 0.0),
        enabled=cfg.get("effect_toggles"),
        seed=cfg.get("seed"),
    )
