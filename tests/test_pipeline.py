"""End-to-end pipeline tests using the synthetic dataset."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from qci.counting import build_counter
from qci.data import build_dataset
from qci.eval.metrics import absolute_error
from qci.optics.degradation import DegradationSim
from qci.optics.encoder import OpticalEncoder
from qci.recovery import build_restoration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dataset():
    return build_dataset(
        {"name": "synthetic", "n_synthetic": 5, "image_size": (64, 64), "max_count": 20, "seed": 0}
    )


@pytest.fixture(scope="module")
def hog_counter():
    return build_counter({"mode": "hog"})


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def test_synthetic_dataset_length(small_dataset):
    assert len(small_dataset) == 5


def test_synthetic_dataset_shapes(small_dataset):
    image, count, density = small_dataset[0]
    assert image.shape == (3, 64, 64)
    assert density.shape == (1, 64, 64)
    assert isinstance(count, int)
    assert count > 0


def test_synthetic_density_integrates_to_count(small_dataset):
    for i in range(len(small_dataset)):
        _, count, density = small_dataset[i]
        assert abs(density.sum().item() - count) < 0.5, (
            f"Sample {i}: density sum {density.sum():.2f} ≠ count {count}"
        )


# ---------------------------------------------------------------------------
# Forward pass: encode -> recover -> count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_forward_pass_shape(small_dataset, hog_counter, strength):
    enc = OpticalEncoder(mode="defocus", strength=strength, kernel_size=11)
    rec = build_restoration({"mode": "none"})

    image, _, _ = small_dataset[0]
    x = image.unsqueeze(0)  # (1,3,H,W)
    y = enc(x)
    y_rec = rec(y)
    counts = hog_counter(y_rec)

    assert counts.shape == (1,)
    assert counts[0].item() >= 0


# ---------------------------------------------------------------------------
# Recovery modules: output shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "unet"])
def test_restoration_shape(mode):
    rec = build_restoration({"mode": mode})
    x = torch.rand(1, 3, 64, 64)
    y = rec(x)
    assert y.shape == x.shape


def test_wiener_restoration_shape():
    enc = OpticalEncoder(mode="defocus", strength=0.5, kernel_size=11)
    psf = enc.get_psf_tensor()
    rec = build_restoration({"mode": "wiener", "nsr": 0.01, "psf": psf})
    x = torch.rand(1, 3, 64, 64)
    y = rec(x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Full sweep mini-run (2 strength values, no GPU, no ShanghaiTech)
# ---------------------------------------------------------------------------


def test_mini_sweep(small_dataset, hog_counter):
    enc = OpticalEncoder(mode="defocus", strength=0.0, kernel_size=11)
    rec = build_restoration({"mode": "none"})
    loader = DataLoader(small_dataset, batch_size=2, shuffle=False)

    results = []
    for strength in [0.0, 0.5]:
        enc.set_strength(strength)
        for images, gt_counts, _ in loader:
            y = enc(images)
            y_rec = rec(y)
            preds = hog_counter(y_rec)
            for pred, gt in zip(preds.tolist(), gt_counts):
                results.append({"strength": strength, "ae": absolute_error(pred, float(gt))})

    assert len(results) == 2 * len(small_dataset)
    # All AE values should be non-negative
    assert all(r["ae"] >= 0 for r in results)


# ---------------------------------------------------------------------------
# Degradation sim integration
# ---------------------------------------------------------------------------


def test_degrade_and_count(small_dataset, hog_counter):
    enc = OpticalEncoder(mode="defocus", strength=0.3, kernel_size=11)
    deg = DegradationSim(gaussian_noise=0.1, seed=99)
    rec = build_restoration({"mode": "none"})

    image, _, _ = small_dataset[0]
    x = image.unsqueeze(0)
    y = enc(x)
    y_d = deg(y)
    y_rec = rec(y_d)
    counts = hog_counter(y_rec)

    assert counts.shape == (1,)
    assert not torch.isnan(counts).any()
