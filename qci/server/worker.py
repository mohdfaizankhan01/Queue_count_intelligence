"""Inference worker — runs the full pipeline on one uploaded image.

All heavy ML work is in ``process_image_bytes`` which is meant to be called
via ``asyncio.to_thread`` so it does not block the event loop.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
import torch

from qci.analytics import ServiceRateModel, QueueStatus
from qci.server.config import ServerConfig

log = logging.getLogger(__name__)


def process_image_bytes(
    station_id: str,
    image_bytes: bytes,
    cfg: ServerConfig,
) -> QueueStatus:
    """Full inference pipeline for one camera frame.

    Steps
    -----
    1. Decode JPEG/PNG → float32 tensor (1, 3, H, W)
    2. Apply OpticalEncoder at ``cfg.encoding_strength``
    3. (Optionally) apply DegradationSim — inference mode, no aug
    4. Run configured CrowdCounter
    5. Estimate queue length (0.5 m per person heuristic)
    6. Run ServiceRateModel
    7. Build and return QueueStatus

    Parameters
    ----------
    station_id:  Station identifier for the resulting QueueStatus.
    image_bytes: Raw bytes of an uploaded image file.
    cfg:         Current ServerConfig.

    Returns
    -------
    QueueStatus
    """
    # ------------------------------------------------------------------
    # 1. Decode
    # ------------------------------------------------------------------
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes — unsupported format?")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)

    # ------------------------------------------------------------------
    # 2. Optical encode
    # ------------------------------------------------------------------
    from qci.optics.encoder import OpticalEncoder

    encoder = OpticalEncoder(mode="defocus", strength=cfg.encoding_strength, kernel_size=11)
    with torch.no_grad():
        encoded = encoder(tensor)

    # ------------------------------------------------------------------
    # 4. Crowd count
    # ------------------------------------------------------------------
    from qci.counting import build_counter

    counter = build_counter({"mode": cfg.counter_type})
    with torch.no_grad():
        count_t = counter(encoded)
    count = float(count_t.squeeze().mean().item())
    count = max(0.0, count)

    # ------------------------------------------------------------------
    # 5–7. Analytics
    # ------------------------------------------------------------------
    queue_length_m = count * 0.5  # rough: ~0.5 m lateral spacing per person
    model = ServiceRateModel(
        n_booths=cfg.n_booths_default,
        avg_service_time_sec=cfg.avg_service_time_sec,
    )
    wait = model.estimate_wait(count)

    status = QueueStatus.create(
        station_id=station_id,
        person_count=count,
        queue_length_m=queue_length_m,
        crowd_density=0.0,
        wait_estimate=wait,
        encoding_strength=cfg.encoding_strength,
        notes="image_upload",
    )
    log.info("Processed image for %s: count=%.0f wait=%.1f min", station_id, count, wait.expected_min)
    return status
