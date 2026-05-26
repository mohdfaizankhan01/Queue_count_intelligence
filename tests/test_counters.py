"""Layer 3 counter tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qci.counting import (
    HOGCounter,
    YOLOCounter,
    build_counter,
    crowd_regime,
)
from qci.counting.csrnet import CSRNetCounter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_batch(b: int = 2, h: int = 128, w: int = 128) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.rand(b, 3, h, w)


# ---------------------------------------------------------------------------
# HOGCounter
# ---------------------------------------------------------------------------


class TestHOGCounter:
    def test_forward_non_negative(self):
        hog = HOGCounter()
        counts = hog(_rand_batch())
        assert (counts >= 0).all()

    def test_forward_shape(self):
        hog = HOGCounter()
        counts = hog(_rand_batch(b=3))
        assert counts.shape == (3,)

    def test_detect_returns_boxes(self):
        hog = HOGCounter()
        x = _rand_batch(b=2, h=480, w=640)
        counts, boxes = hog.detect(x)
        assert counts.shape == (2,)
        assert len(boxes) == 2
        for b in boxes:
            assert b.ndim == 2
            assert b.shape[1] == 4  # xyxy

    def test_detect_matches_forward(self):
        hog = HOGCounter()
        x = _rand_batch(b=2, h=480, w=640)
        fwd_counts = hog(x)
        det_counts, _ = hog.detect(x)
        assert torch.allclose(fwd_counts, det_counts)

    def test_counts_are_integers(self):
        """HOG returns exact detection counts — always whole numbers."""
        hog = HOGCounter()
        counts = hog(_rand_batch(b=3, h=480, w=640))
        for c in counts.tolist():
            assert c == int(c)


# ---------------------------------------------------------------------------
# YOLOCounter — must not crash even without ultralytics installed
# ---------------------------------------------------------------------------


class TestYOLOCounter:
    def test_forward_non_negative(self):
        yolo = YOLOCounter()
        counts = yolo(_rand_batch())
        assert (counts >= 0).all()

    def test_forward_shape(self):
        yolo = YOLOCounter()
        counts = yolo(_rand_batch(b=3))
        assert counts.shape == (3,)

    def test_detect_returns_two_outputs(self):
        yolo = YOLOCounter()
        counts, boxes = yolo.detect(_rand_batch(b=2))
        assert counts.shape == (2,)
        assert len(boxes) == 2

    def test_boxes_xyxy_format(self):
        yolo = YOLOCounter()
        _, boxes = yolo.detect(_rand_batch(b=1, h=480, w=640))
        for b in boxes:
            assert b.ndim == 2
            if len(b) > 0:
                # x1 < x2, y1 < y2
                assert (b[:, 2] >= b[:, 0]).all()
                assert (b[:, 3] >= b[:, 1]).all()


# ---------------------------------------------------------------------------
# CSRNetCounter
# ---------------------------------------------------------------------------


class TestCSRNetCounter:
    def test_forward_non_negative(self):
        net = CSRNetCounter(pretrained_frontend=False, device="cpu")
        counts = net(_rand_batch())
        assert (counts >= 0).all()

    def test_forward_shape(self):
        net = CSRNetCounter(pretrained_frontend=False, device="cpu")
        counts = net(_rand_batch(b=2))
        assert counts.shape == (2,)

    def test_density_map_non_negative(self):
        net = CSRNetCounter(pretrained_frontend=False, device="cpu")
        x = _rand_batch(b=1)
        dmap = net.predict_density_map(x)
        assert (dmap >= -1e-6).all(), "Density map has negative values"

    def test_density_map_shape(self):
        net = CSRNetCounter(pretrained_frontend=False, device="cpu")
        x = _rand_batch(b=2, h=64, w=64)
        dmap = net.predict_density_map(x)
        assert dmap.shape == (2, 1, 64, 64)

    def test_count_scales_with_resolution(self):
        """Counts should be consistent regardless of input spatial size."""
        net = CSRNetCounter(pretrained_frontend=False, device="cpu")
        torch.manual_seed(0)
        x_small = torch.rand(1, 3, 64, 64)
        x_large = torch.rand(1, 3, 128, 128)
        c_small = net(x_small)
        c_large = net(x_large)
        # Both should be finite and non-negative
        assert c_small.isfinite().all()
        assert c_large.isfinite().all()


# ---------------------------------------------------------------------------
# build_counter factory
# ---------------------------------------------------------------------------


class TestCounterFactory:
    def test_hog(self):
        c = build_counter({"mode": "hog"})
        assert isinstance(c, HOGCounter)

    def test_yolo(self):
        c = build_counter({"mode": "yolo"})
        assert isinstance(c, YOLOCounter)

    def test_density(self):
        c = build_counter({"mode": "density", "pretrained_frontend": False})
        assert isinstance(c, CSRNetCounter)

    def test_csrnet_alias(self):
        c = build_counter({"mode": "csrnet", "pretrained_frontend": False})
        assert isinstance(c, CSRNetCounter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown counter mode"):
            build_counter({"mode": "magic_counter"})


# ---------------------------------------------------------------------------
# crowd_regime
# ---------------------------------------------------------------------------


class TestCrowdRegime:
    def test_sparse_boundary(self):
        assert crowd_regime(0) == "sparse"
        assert crowd_regime(19.9) == "sparse"

    def test_medium_boundary(self):
        assert crowd_regime(20) == "medium"
        assert crowd_regime(80) == "medium"

    def test_dense_boundary(self):
        assert crowd_regime(80.1) == "dense"
        assert crowd_regime(500) == "dense"

    def test_exact_boundaries(self):
        assert crowd_regime(20) == "medium"
        assert crowd_regime(80) == "medium"
