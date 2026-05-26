"""PSFInversionAttacker — measures how well an attacker can undo optical encoding.

Two attack strategies:
- Wiener oracle: attacker knows the exact PSF (upper bound on attack power).
- Richardson–Lucy blind: attacker knows only the kernel size and guesses
  a Gaussian PSF (realistic blind deconvolution).

Quality metric: PSNR between the inverted image and the original clear image.
Higher PSNR = better recovery = worse privacy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deconvolution helpers
# ---------------------------------------------------------------------------

def _wiener_deconv(encoded_ch: np.ndarray, psf: np.ndarray, nsr: float = 0.01) -> np.ndarray:
    """Wiener deconvolution for a single channel (float32, [0,1])."""
    H = np.fft.fft2(psf, s=encoded_ch.shape)
    Y = np.fft.fft2(encoded_ch)
    Hc = np.conj(H)
    W = Hc / (np.abs(H) ** 2 + nsr)
    x_hat = np.real(np.fft.ifft2(W * Y))
    return np.clip(x_hat, 0.0, 1.0).astype(np.float32)


def _richardson_lucy(
    encoded_ch: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 30,
) -> np.ndarray:
    """Richardson–Lucy blind deconvolution for one channel."""
    from scipy.signal import fftconvolve  # type: ignore

    psf_norm = psf / (psf.sum() + 1e-12)
    psf_flip = np.flipud(np.fliplr(psf_norm))

    x = np.full_like(encoded_ch, 0.5, dtype=np.float64)
    y = encoded_ch.astype(np.float64)
    for _ in range(n_iter):
        conv = fftconvolve(x, psf_norm, mode="same")
        conv = np.maximum(conv, 1e-12)
        ratio = y / conv
        x = x * fftconvolve(ratio, psf_flip, mode="same")
        x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)


def _psnr(original: np.ndarray, recovered: np.ndarray) -> float:
    """PSNR in dB; images assumed float32 in [0, 1]."""
    mse = float(np.mean((original.astype(np.float64) - recovered.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """2-D Gaussian kernel (normalised)."""
    half = size // 2
    xs = np.arange(-half, half + 1)
    g = np.exp(-(xs**2) / (2 * sigma**2))
    kernel = np.outer(g, g)
    return (kernel / kernel.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InversionResult:
    """Inversion quality for one strength level."""

    strength: float
    psnr_wiener: float          # Oracle attack (knows exact PSF)
    psnr_rl: float              # Blind Richardson–Lucy attack
    # Example (original, encoded, wiener_recovered, rl_recovered) tuples
    examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Main attacker
# ---------------------------------------------------------------------------

class PSFInversionAttacker:
    """Measure PSF inversion quality after optical encoding.

    Parameters
    ----------
    n_rl_iter:
        Number of Richardson–Lucy iterations.
    wiener_nsr:
        Noise-to-signal ratio for Wiener filter.
    n_examples:
        How many (original, encoded, recovered) example tuples to keep
        per strength level (for visualisation).
    """

    def __init__(
        self,
        n_rl_iter: int = 30,
        wiener_nsr: float = 0.01,
        n_examples: int = 3,
    ) -> None:
        self.n_rl_iter = n_rl_iter
        self.wiener_nsr = wiener_nsr
        self.n_examples = n_examples

    # ------------------------------------------------------------------

    def _encode_images(
        self,
        images: np.ndarray,
        encoder,
        strength: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode images; also return the PSF tensor from the encoder.

        Returns (encoded_images, psf_numpy).
        encoded_images: (N, H, W, 3) float32
        psf_numpy: (kH, kW) float32
        """
        encoder.set_strength(strength)
        encoded = []
        psf_np = None
        for img in images:
            t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                enc = encoder(t)
                if psf_np is None:
                    psf_t = encoder.get_psf_tensor()
                    psf_np = psf_t.squeeze().cpu().numpy()
            arr = enc.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            encoded.append(arr)

        if psf_np is None:
            psf_np = _gaussian_kernel(11, 2.0)

        return np.stack(encoded, axis=0), psf_np

    def _invert_image(
        self,
        encoded: np.ndarray,
        psf_oracle: np.ndarray,
        psf_blind: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Invert a single (H, W, 3) encoded image.

        Returns (wiener_recovered, rl_recovered).
        """
        wiener_ch = []
        rl_ch = []
        for c in range(encoded.shape[2]):
            wiener_ch.append(_wiener_deconv(encoded[..., c], psf_oracle, self.wiener_nsr))
            rl_ch.append(_richardson_lucy(encoded[..., c], psf_blind, self.n_rl_iter))
        wiener_rec = np.stack(wiener_ch, axis=-1)
        rl_rec = np.stack(rl_ch, axis=-1)
        return wiener_rec, rl_rec

    def evaluate(
        self,
        images: np.ndarray,
        encoder,
        strength: float,
    ) -> InversionResult:
        """Compute Wiener and RL PSNR for one strength level.

        images: (N, H, W, 3) float32 in [0, 1]
        """
        encoded_imgs, psf_oracle = self._encode_images(images, encoder, strength)

        # Blind attacker assumes Gaussian with same kernel footprint
        ksize = max(psf_oracle.shape)
        sigma_blind = max(ksize / 6.0, 0.5)
        psf_blind = _gaussian_kernel(ksize, sigma_blind)

        psnr_w_list = []
        psnr_rl_list = []
        examples = []

        for i, (orig, enc) in enumerate(zip(images, encoded_imgs)):
            w_rec, rl_rec = self._invert_image(enc, psf_oracle, psf_blind)
            psnr_w_list.append(_psnr(orig, w_rec))
            psnr_rl_list.append(_psnr(orig, rl_rec))
            if i < self.n_examples:
                examples.append((orig.copy(), enc.copy(), w_rec, rl_rec))

        psnr_w = float(np.mean(psnr_w_list))
        psnr_rl = float(np.mean(psnr_rl_list))
        log.info(
            "strength=%.2f  PSNR_wiener=%.1f dB  PSNR_RL=%.1f dB",
            strength, psnr_w, psnr_rl,
        )
        return InversionResult(
            strength=strength,
            psnr_wiener=psnr_w,
            psnr_rl=psnr_rl,
            examples=examples,
        )

    def sweep(
        self,
        images: np.ndarray,
        encoder,
        strengths: List[float],
    ) -> List[InversionResult]:
        """Run inversion evaluation across multiple encoding strengths."""
        return [self.evaluate(images, encoder, s) for s in strengths]
