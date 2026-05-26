from .base import RestorationModule
from .none import IdentityRestoration
from .classical import WienerRestoration, UNetStub


def build_restoration(cfg: dict) -> RestorationModule:
    """Factory: return a RestorationModule from config."""
    mode = cfg.get("mode", "none")
    if mode == "none":
        return IdentityRestoration()
    elif mode == "wiener":
        return WienerRestoration(
            psf=cfg.get("psf"),          # (1,1,kH,kW) tensor or None
            nsr=cfg.get("nsr", 0.01),
        )
    elif mode == "unet":
        return UNetStub(
            in_channels=cfg.get("in_channels", 3),
            features=cfg.get("features", 32),
        )
    else:
        raise ValueError(f"Unknown restoration mode: {mode!r}")


__all__ = [
    "RestorationModule",
    "IdentityRestoration",
    "WienerRestoration",
    "UNetStub",
    "build_restoration",
]
