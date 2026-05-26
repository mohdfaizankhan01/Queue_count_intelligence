"""FaceRecognitionAttacker — measures Equal Error Rate after optical encoding.

Uses HOG feature vectors (always available via scikit-image or skimage-like
reimplementation) as face embeddings.  Cosine similarity drives the
verification decision.

EER: the threshold at which FAR = FRR.
  FAR (False Accept Rate) = same-person pairs below threshold / total impostor
  FRR (False Reject Rate) = different-person pairs above threshold / total genuine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HOG embedding helpers
# ---------------------------------------------------------------------------

def _hog_embedding(img_float: np.ndarray, pixels_per_cell: int = 8) -> np.ndarray:
    """Return a L2-normalised HOG descriptor for an (H, W, 3) float32 image."""
    try:
        from skimage.feature import hog as sk_hog  # type: ignore
        gray = 0.299 * img_float[..., 0] + 0.587 * img_float[..., 1] + 0.114 * img_float[..., 2]
        desc = sk_hog(
            gray,
            orientations=9,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
    except ImportError:
        # Minimal fallback: gradient magnitude histogram per channel
        gray = (0.299 * img_float[..., 0] + 0.587 * img_float[..., 1] + 0.114 * img_float[..., 2])
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        desc = np.histogram(mag.ravel(), bins=72, range=(0, 1))[0].astype(np.float32)

    norm = np.linalg.norm(desc)
    return desc / (norm + 1e-9)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# EER computation
# ---------------------------------------------------------------------------

def _compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute EER by sweeping thresholds.

    Returns (eer, threshold).
    """
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.5, 0.0

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.sort(np.unique(all_scores))

    best_diff = float("inf")
    best_eer = 0.5
    best_thr = 0.0
    for thr in thresholds:
        # Accept if score >= threshold
        frr = float((genuine_scores < thr).mean())   # genuine rejected
        far = float((impostor_scores >= thr).mean())  # impostors accepted
        diff = abs(frr - far)
        if diff < best_diff:
            best_diff = diff
            best_eer = (frr + far) / 2.0
            best_thr = thr

    return best_eer, best_thr


# ---------------------------------------------------------------------------
# Main attacker
# ---------------------------------------------------------------------------

@dataclass
class RecognitionResult:
    """Verification result for one strength level."""

    strength: float
    eer: float
    threshold: float
    n_genuine: int
    n_impostor: int


class FaceRecognitionAttacker:
    """Evaluate face verification performance after encoding.

    Build-a-gallery strategy:
    - For each identity, the first image is the gallery reference.
    - Remaining images are probes.
    - Genuine pairs: probe vs gallery of same identity.
    - Impostor pairs: probe vs gallery of different identity (sampled).

    Parameters
    ----------
    max_impostor_pairs:
        Cap on impostor pair comparisons (keeps runtime manageable).
    pixels_per_cell:
        HOG cell size.  Smaller = richer but slower.
    """

    def __init__(
        self,
        max_impostor_pairs: int = 2000,
        pixels_per_cell: int = 8,
    ) -> None:
        self.max_impostor_pairs = max_impostor_pairs
        self.pixels_per_cell = pixels_per_cell

    # ------------------------------------------------------------------

    def _encode_images(
        self,
        images: np.ndarray,
        encoder,
        strength: float,
    ) -> np.ndarray:
        """Encode images (N, H, W, 3) float32 → (N, H, W, 3) float32."""
        encoder.set_strength(strength)
        encoded = []
        for img in images:
            t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                enc = encoder(t)
            arr = enc.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            encoded.append(arr)
        return np.stack(encoded, axis=0)

    def _build_embeddings(self, images: np.ndarray) -> np.ndarray:
        """Return (N,) array of HOG embeddings."""
        return np.stack([_hog_embedding(img, self.pixels_per_cell) for img in images], axis=0)

    def _make_pairs(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build genuine and impostor cosine similarity arrays."""
        n = len(embeddings)
        rng = np.random.default_rng(0)

        genuine_scores = []
        impostor_scores = []

        # Genuine: same-label pairs
        for pid in np.unique(labels):
            idx = np.where(labels == pid)[0]
            if len(idx) < 2:
                continue
            anchor = embeddings[idx[0]]
            for i in idx[1:]:
                genuine_scores.append(_cosine_sim(anchor, embeddings[i]))

        # Impostor: cross-label pairs, capped
        all_idx = np.arange(n)
        n_imp = min(self.max_impostor_pairs, n * (n - 1) // 2)
        sampled = 0
        attempts = 0
        while sampled < n_imp and attempts < n_imp * 10:
            i, j = rng.choice(all_idx, 2, replace=False)
            if labels[i] != labels[j]:
                impostor_scores.append(_cosine_sim(embeddings[i], embeddings[j]))
                sampled += 1
            attempts += 1

        return np.array(genuine_scores, dtype=np.float32), np.array(impostor_scores, dtype=np.float32)

    def evaluate(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        encoder,
        strength: float,
    ) -> RecognitionResult:
        """Compute EER at one strength level."""
        encoded = self._encode_images(images, encoder, strength)
        embeddings = self._build_embeddings(encoded)
        genuine, impostor = self._make_pairs(embeddings, labels)
        eer, thr = _compute_eer(genuine, impostor)
        log.info(
            "strength=%.2f  EER=%.3f  genuine=%d  impostor=%d",
            strength, eer, len(genuine), len(impostor),
        )
        return RecognitionResult(
            strength=strength,
            eer=eer,
            threshold=thr,
            n_genuine=len(genuine),
            n_impostor=len(impostor),
        )

    def sweep(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        encoder,
        strengths: List[float],
    ) -> List[RecognitionResult]:
        """Run EER evaluation across multiple encoding strengths."""
        return [self.evaluate(images, labels, encoder, s) for s in strengths]
