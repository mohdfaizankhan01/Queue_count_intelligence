from .base import CrowdCounter
from .csrnet import CSRNetCounter
from .density import DensityCounter          # backward-compat alias
from .hog import HOGCounter
from .yolo import YOLOCounter
from .crowd_regime import crowd_regime, REGIMES


def build_counter(cfg: dict) -> CrowdCounter:
    """Factory: construct a ``CrowdCounter`` from a config dict.

    ``cfg["mode"]`` selects the implementation:

    * ``"hog"``     — OpenCV HOG + SVM detector (no training required)
    * ``"yolo"``    — YOLOv8n person detector (requires ``ultralytics``)
    * ``"density"`` or ``"csrnet"`` — CSRNet density-map counter
    """
    mode = cfg.get("mode", "hog")

    if mode == "hog":
        return HOGCounter(
            win_stride=tuple(cfg.get("win_stride", [8, 8])),
            scale=cfg.get("scale", 1.05),
            padding=tuple(cfg.get("padding", [8, 8])),
        )
    elif mode == "yolo":
        return YOLOCounter(
            model_name=cfg.get("model_name", "yolov8n.pt"),
            confidence_threshold=cfg.get("confidence_threshold", 0.4),
            device=cfg.get("device"),
        )
    elif mode in ("density", "csrnet"):
        return CSRNetCounter(
            pretrained_path=cfg.get("pretrained_path"),
            weights_url=cfg.get("weights_url"),
            pretrained_frontend=cfg.get("pretrained_frontend", True),
            device=cfg.get("device", "cpu"),
        )
    else:
        raise ValueError(
            f"Unknown counter mode: {mode!r}. "
            "Valid options: 'hog', 'yolo', 'density', 'csrnet'."
        )


__all__ = [
    "CrowdCounter",
    "CSRNetCounter",
    "DensityCounter",
    "HOGCounter",
    "YOLOCounter",
    "crowd_regime",
    "REGIMES",
    "build_counter",
]
