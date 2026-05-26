"""Tests for DegradationSim (Layer 1)."""

import pytest
import torch

from qci.optics.degradation import DegradationSim


def _batch(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(2, 3, 64, 64)


# ---------------------------------------------------------------------------
# Severity-0 effects are identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "effect,kwargs",
    [
        ("motion_blur",     {"motion_blur": 0.0}),
        ("low_light",       {"low_light": 0.0}),
        ("gaussian_noise",  {"gaussian_noise": 0.0}),
        ("downsample",      {"downsample": 0.0}),
    ],
)
def test_zero_severity_is_identity(effect, kwargs):
    sim = DegradationSim(**kwargs, seed=42)
    x = _batch()
    y = sim(x)
    assert torch.allclose(y, x, atol=1e-5), f"{effect} with severity=0 changed the image"


# ---------------------------------------------------------------------------
# Output shape preserved for every effect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gaussian_noise": 0.5},
        {"poisson_noise": 0.5},
        {"motion_blur": 0.5},
        {"low_light": 0.5},
        {"downsample": 0.5},
    ],
)
def test_output_shape(kwargs):
    sim = DegradationSim(**kwargs, seed=42)
    x = _batch()
    y = sim(x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Output values remain in [0, 1]
# ---------------------------------------------------------------------------


def test_all_effects_clamp():
    sim = DegradationSim(
        gaussian_noise=0.8,
        poisson_noise=0.8,
        motion_blur=0.8,
        low_light=0.8,
        downsample=0.8,
        seed=0,
    )
    x = _batch()
    y = sim(x)
    assert y.min() >= -1e-5
    assert y.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Seeded noise is reproducible
# ---------------------------------------------------------------------------


def test_seeded_gaussian_reproducible():
    sim1 = DegradationSim(gaussian_noise=0.3, seed=7)
    sim2 = DegradationSim(gaussian_noise=0.3, seed=7)
    x = _batch()
    assert torch.allclose(sim1(x), sim2(x))


# ---------------------------------------------------------------------------
# High severity makes image darker (low_light)
# ---------------------------------------------------------------------------


def test_low_light_darkens():
    sim_dark = DegradationSim(low_light=0.9)
    sim_none = DegradationSim(low_light=0.0)
    x = _batch()
    assert sim_dark(x).mean() < sim_none(x).mean()


# ---------------------------------------------------------------------------
# Downsample reduces high-frequency content
# ---------------------------------------------------------------------------


def test_downsample_blurs():
    """Heavy downsampling should reduce pixel-level variance."""
    sim = DegradationSim(downsample=0.9)
    x = _batch()
    assert sim(x).std() <= x.std() + 0.05
