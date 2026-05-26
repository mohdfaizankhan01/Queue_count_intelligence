"""CSRNet — density-map crowd counter (Li et al., CVPR 2018).

Architecture:
  * Front-end : VGG-16 conv1_1 → pool3  (pretrained ImageNet; stride 8)
  * Back-end  : 6 dilated-conv layers (dilation 2) + 1×1 output conv
  * Count     : density_map.sum() × (H×W) / (map_H×map_W)

Weight loading priority:
  1. ``pretrained_path`` — local .pth/.pt file (state_dict)
  2. ``weights_url``     — remote URL (attempted via torch.hub.download)
  3. Random back-end     — warns that fine-tuning is needed
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional
from urllib.error import URLError

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CrowdCounter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Front-end
# ---------------------------------------------------------------------------


def _build_vgg_frontend(pretrained: bool = True) -> nn.Sequential:
    """VGG-16 conv1_1 → conv4_3 (features[0:23], output stride 8, 512 ch).

    features[:23] = 10 conv layers (conv1–conv4) + 3 pooling layers + ReLUs.
    After pool3 spatial stride is 8; conv4_1–conv4_3 keep stride 8 and lift
    channels from 256 → 512.  This matches the original CSRNet paper front-end.
    """
    try:
        import torchvision.models as tvm
        weights = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = tvm.vgg16(weights=weights)
        frontend = nn.Sequential(*list(vgg.features.children())[:23])
        if pretrained:
            logger.info("CSRNet: loaded VGG-16 ImageNet front-end (512-ch output)")
        return frontend
    except Exception as exc:
        warnings.warn(f"VGG-16 unavailable ({exc}); using lightweight substitute.")
        return _build_lightweight_frontend()


def _build_lightweight_frontend() -> nn.Sequential:
    """Lightweight substitute matching VGG-16 front-end output (512 ch, stride 8)."""
    return nn.Sequential(
        nn.Conv2d(3,   64,  3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(64,  64,  3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
    )  # output: (B, 512, H/8, W/8)


# ---------------------------------------------------------------------------
# Back-end (6 dilated layers)
# ---------------------------------------------------------------------------


def _build_backend() -> nn.Sequential:
    """Six dilated-conv layers → 1-channel density map.

    Accepts 512-channel front-end output (VGG conv4_3 / lightweight substitute).
    """
    return nn.Sequential(
        nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d(128,  64, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d( 64,  64, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
        nn.Conv2d( 64,   1, 1),
        nn.ReLU(inplace=True),  # density values are non-negative
    )


# ---------------------------------------------------------------------------
# Weight downloader
# ---------------------------------------------------------------------------


def _try_download_weights(url: str, dst_path: Path, map_location: str) -> bool:
    """Download pretrained weights from *url* to *dst_path*.  Returns success."""
    import urllib.request

    try:
        logger.info(f"Downloading CSRNet weights from {url} …")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dst_path))
        logger.info(f"Saved to {dst_path}")
        return True
    except (URLError, OSError) as exc:
        logger.warning(f"Weight download failed ({exc}); using random back-end.")
        if dst_path.exists():
            dst_path.unlink()
        return False


# ---------------------------------------------------------------------------
# CSRNetCounter
# ---------------------------------------------------------------------------


class CSRNetCounter(CrowdCounter):
    """CSRNet crowd counter.

    Predicted count = density_map.sum() × (H×W / map_H×map_W)
    so the scaling adapts to arbitrary input resolutions.
    """

    # Known public weight URLs (ShanghaiTech Part B).  Update if URLs change.
    _DEFAULT_WEIGHTS_URL: Optional[str] = None  # no canonical CDN available yet
    _DEFAULT_WEIGHTS_CACHE = Path.home() / ".cache" / "qci" / "csrnet_partb.pth"

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        weights_url: Optional[str] = None,
        pretrained_frontend: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device_str = device

        self.frontend = _build_vgg_frontend(pretrained=pretrained_frontend)
        self.backend = _build_backend()

        for param in self.frontend.parameters():
            param.requires_grad_(False)

        # --- weight loading ---
        loaded = False
        if pretrained_path is not None:
            p = Path(pretrained_path)
            if p.exists():
                state = torch.load(str(p), map_location=device, weights_only=True)
                self.load_state_dict(state, strict=False)
                logger.info(f"Loaded CSRNet weights from {p}")
                loaded = True
            else:
                warnings.warn(f"pretrained_path {p} not found.")

        if not loaded:
            eff_url = weights_url or self._DEFAULT_WEIGHTS_URL
            if eff_url and not self._DEFAULT_WEIGHTS_CACHE.exists():
                loaded = _try_download_weights(eff_url, self._DEFAULT_WEIGHTS_CACHE, device)
                if loaded:
                    state = torch.load(str(self._DEFAULT_WEIGHTS_CACHE), map_location=device, weights_only=True)
                    self.load_state_dict(state, strict=False)
            elif self._DEFAULT_WEIGHTS_CACHE.exists():
                state = torch.load(str(self._DEFAULT_WEIGHTS_CACHE), map_location=device, weights_only=True)
                self.load_state_dict(state, strict=False)
                loaded = True

        if not loaded:
            warnings.warn(
                "CSRNet back-end is randomly initialised. "
                "Counts will be uninformative until fine-tuned on ShanghaiTech."
            )

        self.to(device)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B,)`` predicted counts."""
        B, C, H, W = x.shape
        x_dev = x.to(self.device_str)
        feat = self.frontend(x_dev)
        dmap = self.backend(feat)                    # (B, 1, map_H, map_W)
        map_H, map_W = dmap.shape[2], dmap.shape[3]
        scale = (H * W) / max(map_H * map_W, 1)
        counts = dmap.sum(dim=(1, 2, 3)) * scale
        return counts.cpu()

    def predict_density_map(self, x: torch.Tensor) -> torch.Tensor:
        """Upsampled density map ``(B, 1, H, W)`` for visualisation."""
        B, C, H, W = x.shape
        x_dev = x.to(self.device_str)
        feat = self.frontend(x_dev)
        dmap = self.backend(feat)
        return F.interpolate(dmap, size=(H, W), mode="bilinear", align_corners=False).cpu()
