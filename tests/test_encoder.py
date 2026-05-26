"""Tests for OpticalEncoder (Layer 1)."""

import pytest
import torch

from qci.optics.encoder import OpticalEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _batch(b=2, c=3, h=64, w=64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(b, c, h, w)


# ---------------------------------------------------------------------------
# Identity at strength=0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["defocus", "coded_mask", "learnable_psf"])
def test_identity_at_zero_strength(mode):
    enc = OpticalEncoder(mode=mode, strength=0.0, kernel_size=11)
    x = _batch()
    y = enc(x)
    assert torch.allclose(y, x, atol=1e-5), f"mode={mode}: expected identity at strength=0"


# ---------------------------------------------------------------------------
# Output shape is preserved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["defocus", "coded_mask", "learnable_psf"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (3, 3, 64, 128)])
def test_output_shape(mode, shape):
    enc = OpticalEncoder(mode=mode, strength=0.5, kernel_size=11)
    x = torch.rand(*shape)
    y = enc(x)
    assert y.shape == x.shape, f"mode={mode}: shape mismatch {y.shape} vs {x.shape}"


# ---------------------------------------------------------------------------
# Output values are in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["defocus", "coded_mask", "learnable_psf"])
def test_output_range(mode):
    enc = OpticalEncoder(mode=mode, strength=0.8, kernel_size=11)
    x = _batch()
    y = enc(x)
    assert y.min() >= -1e-6 and y.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Gradients flow through learnable_psf
# ---------------------------------------------------------------------------


def test_learnable_psf_gradients():
    enc = OpticalEncoder(mode="learnable_psf", strength=0.7, kernel_size=11)
    x = _batch(b=1)
    y = enc(x)
    loss = y.sum()
    loss.backward()
    assert enc._learnable_psf.grad is not None, "No gradient on _learnable_psf"
    assert not torch.isnan(enc._learnable_psf.grad).any(), "NaN gradient"


# ---------------------------------------------------------------------------
# Energy conservation (PSF sums to 1 → uniform input preserved)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["defocus", "coded_mask"])
def test_energy_conservation(mode):
    """PSF kernel weights must sum to 1 (energy conservation).

    Testing the kernel sum rather than image-level energy because zero-padding
    correctly leaks ~5–30 % of energy near image borders — that is physically
    accurate, not a bug.
    """
    enc = OpticalEncoder(mode=mode, strength=0.5, kernel_size=11)
    psf = enc.get_psf_tensor()
    assert abs(psf.sum().item() - 1.0) < 1e-4, (
        f"mode={mode}: PSF does not sum to 1; got {psf.sum().item():.6f}"
    )
    # Interior pixels of a large uniform image should be close to input value
    x = torch.ones(1, 1, 256, 256) * 0.5
    y = enc(x)
    interior = y[0, 0, 20:-20, 20:-20]
    assert abs(interior.mean().item() - 0.5) < 0.01, (
        f"mode={mode}: interior energy not conserved; got mean={interior.mean():.4f}"
    )


# ---------------------------------------------------------------------------
# set_strength API
# ---------------------------------------------------------------------------


def test_set_strength_clip():
    enc = OpticalEncoder(mode="defocus", strength=0.5, kernel_size=11)
    enc.set_strength(1.5)
    assert enc.strength == 1.0
    enc.set_strength(-0.3)
    assert enc.strength == 0.0


# ---------------------------------------------------------------------------
# get_psf_tensor shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["defocus", "coded_mask", "learnable_psf"])
def test_get_psf_tensor_shape(mode):
    enc = OpticalEncoder(mode=mode, strength=0.5, kernel_size=15)
    psf = enc.get_psf_tensor()
    assert psf.shape == (1, 1, 15, 15)


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------


def test_invalid_mode():
    with pytest.raises(ValueError):
        OpticalEncoder(mode="unknown")


def test_even_kernel_size():
    with pytest.raises(ValueError):
        OpticalEncoder(mode="defocus", kernel_size=32)
