"""FaceDetectionAttacker — measures Face Detection Rate after optical encoding."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

# Haar cascade — always available in OpenCV
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# OpenCV DNN face detector weights (download once from the OpenCV model zoo)
_DNN_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_DNN_WEIGHTS = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"


@dataclass
class FaceDetectionResult:
    """Result for one strength level."""

    strength: float
    fdr: float                          # Fraction of faces detected
    detected: int
    total: int
    # Per-image detection counts (optional)
    per_image: List[int] = field(default_factory=list)


class FaceDetectionAttacker:
    """Evaluate how many faces remain detectable after encoding.

    Uses OpenCV Haar cascade (always available) with an optional DNN
    detector fallback for better precision.  The DNN model is loaded
    lazily and silently skipped if unavailable.

    Parameters
    ----------
    use_dnn:
        Whether to also try the DNN detector (requires network access once
        to download the model files, or local paths provided).
    haar_scale: scaleFactor for Haar cascade detectMultiScale.
    haar_min_neighbors: minNeighbors — higher = fewer false positives.
    min_face_size: Minimum face size in pixels.
    """

    def __init__(
        self,
        use_dnn: bool = False,
        haar_scale: float = 1.1,
        haar_min_neighbors: int = 3,
        min_face_size: int = 8,
    ) -> None:
        self.haar_scale = haar_scale
        self.haar_min_neighbors = haar_min_neighbors
        self.min_face_size = (min_face_size, min_face_size)

        self._haar = cv2.CascadeClassifier(_HAAR_PATH)
        if self._haar.empty():
            raise RuntimeError(f"Could not load Haar cascade from {_HAAR_PATH}")

        self._dnn: Optional[cv2.dnn.Net] = None
        if use_dnn:
            self._dnn = self._try_load_dnn()

    # ------------------------------------------------------------------

    def _try_load_dnn(self) -> Optional[cv2.dnn.Net]:
        try:
            import urllib.request
            import tempfile, os

            proto_path = tempfile.mktemp(suffix=".prototxt")
            weights_path = tempfile.mktemp(suffix=".caffemodel")
            urllib.request.urlretrieve(_DNN_PROTO, proto_path)
            urllib.request.urlretrieve(_DNN_WEIGHTS, weights_path)
            net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
            os.unlink(proto_path)
            os.unlink(weights_path)
            log.info("DNN face detector loaded.")
            return net
        except Exception as exc:
            log.warning("DNN face detector unavailable (%s); using Haar only.", exc)
            return None

    # ------------------------------------------------------------------

    def _detect_haar(self, gray: np.ndarray) -> int:
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=self.haar_scale,
            minNeighbors=self.haar_min_neighbors,
            minSize=self.min_face_size,
        )
        return len(faces) if faces is not None and len(faces) > 0 else 0

    def _detect_dnn(self, bgr: np.ndarray, conf_threshold: float = 0.5) -> int:
        if self._dnn is None:
            return 0
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False
        )
        self._dnn.setInput(blob)
        out = self._dnn.forward()
        count = int((out[0, 0, :, 2] >= conf_threshold).sum())
        return count

    def detect_image(self, img_float: np.ndarray) -> int:
        """Detect faces in a float32 (H,W,3) image in [0,1].

        Returns number of faces detected (Haar OR DNN, max of both).
        """
        img_uint8 = (img_float * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        n_haar = self._detect_haar(gray)
        n_dnn = self._detect_dnn(img_uint8) if self._dnn else 0
        return max(n_haar, n_dnn)

    # ------------------------------------------------------------------

    def _encode_images(
        self,
        images: np.ndarray,
        encoder,
        strength: float,
    ) -> np.ndarray:
        """Apply encoder at given strength to all images.

        images: (N, H, W, 3) float32
        Returns: (N, H, W, 3) float32 encoded
        """
        encoder.set_strength(strength)
        encoded = []
        for img in images:
            t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                enc = encoder(t)
            arr = enc.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            encoded.append(arr)
        return np.stack(encoded, axis=0)

    def evaluate(
        self,
        images: np.ndarray,
        encoder,
        strength: float,
    ) -> FaceDetectionResult:
        """Compute FDR for one strength value.

        Each image is assumed to contain exactly one face (standard LFW
        assumption).  FDR = detected / total.
        """
        encoded = self._encode_images(images, encoder, strength)
        per_image = [self.detect_image(img) for img in encoded]
        detected = sum(1 for n in per_image if n >= 1)
        total = len(images)
        fdr = detected / total if total > 0 else 0.0
        return FaceDetectionResult(
            strength=strength,
            fdr=fdr,
            detected=detected,
            total=total,
            per_image=per_image,
        )

    def sweep(
        self,
        images: np.ndarray,
        encoder,
        strengths: List[float],
    ) -> List[FaceDetectionResult]:
        """Run FDR evaluation across multiple encoding strengths."""
        results = []
        for s in strengths:
            r = self.evaluate(images, encoder, s)
            log.info("strength=%.2f  FDR=%.3f  (%d/%d)", s, r.fdr, r.detected, r.total)
            results.append(r)
        return results
