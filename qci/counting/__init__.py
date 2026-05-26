from .base import CrowdCounter
from .blob import BlobCounter
from .csrnet import CSRNetCounter
from .density import DensityCounter          # backward-compat alias
from .hog import HOGCounter
from .yolo import YOLOCounter
from .crowd_regime import crowd_regime, REGIMES


def build_counter(cfg: dict) -> CrowdCounter:
    """Factory: construct a ``CrowdCounter`` from a config dict.

    ``cfg["mode"]`` selects the implementation:

    * ``"blob"``    — background-subtraction blob counter (synthetic data)
    * ``"hog"``     — OpenCV HOG + SVM detector (real person images)
    * ``"yolo"``    — YOLOv8n person detector (requires ``ultralytics``)
    * ``"density"`` or ``"csrnet"`` — CSRNet density-map counter
    """
    mode = cfg.get("mode", "hog")

    if mode == "blob":
        return BlobCounter(
            window_size=cfg.get("window_size", 3),
            std_threshold=cfg.get("std_threshold", 11.0),
            min_blob_area=cfg.get("min_blob_area", 6),
        )
    elif mode == "hog":
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
            "Valid options: 'blob', 'hog', 'yolo', 'density', 'csrnet'."
        )


__all__ = [
    "BlobCounter",
    "CrowdCounter",
    "CSRNetCounter",
    "DensityCounter",
    "HOGCounter",
    "YOLOCounter",
    "crowd_regime",
    "REGIMES",
    "build_counter",
]
