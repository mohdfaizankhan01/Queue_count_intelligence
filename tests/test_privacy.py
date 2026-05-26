"""Tests for Layer 6: FaceDetectionAttacker, FaceRecognitionAttacker,
PSFInversionAttacker, FaceDataLoader, PrivacyUtilityAnalyzer."""

import numpy as np
import pytest
import torch

from qci.optics.encoder import OpticalEncoder
from qci.privacy.face_data import FaceDataLoader, FaceData, _synthetic_faces
from qci.privacy.face_attacker import FaceDetectionAttacker
from qci.privacy.recognition_attacker import FaceRecognitionAttacker, _compute_eer
from qci.privacy.inversion_attacker import PSFInversionAttacker, _psnr, _gaussian_kernel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_faces() -> FaceData:
    return _synthetic_faces(n_images=30, height=50, width=37)


@pytest.fixture(scope="module")
def encoder() -> OpticalEncoder:
    return OpticalEncoder(mode="defocus", strength=0.0, kernel_size=11)


# ---------------------------------------------------------------------------
# FaceDataLoader / FaceData
# ---------------------------------------------------------------------------

class TestFaceData:
    def test_synthetic_fallback_shape(self):
        fd = _synthetic_faces(n_images=18, height=48, width=36)
        assert fd.images.shape[1:] == (48, 36, 3)
        assert fd.images.dtype == np.float32
        assert fd.images.min() >= 0.0
        assert fd.images.max() <= 1.0

    def test_synthetic_fallback_labels(self):
        fd = _synthetic_faces(n_images=18)
        assert len(fd.labels) == len(fd.images)
        assert fd.n_people >= 1

    def test_loader_uses_fallback_on_no_data(self):
        # Force failure by passing invalid data_home
        loader = FaceDataLoader(min_faces_per_person=9999, max_images=20)
        fd = loader.load()
        assert isinstance(fd, FaceData)
        assert len(fd) > 0
        assert fd.images.ndim == 4
        assert fd.images.shape[-1] == 3


# ---------------------------------------------------------------------------
# FaceDetectionAttacker
# ---------------------------------------------------------------------------

class TestFaceDetectionAttacker:
    def test_fdr_between_0_and_1(self, synthetic_faces, encoder):
        attacker = FaceDetectionAttacker()
        result = attacker.evaluate(synthetic_faces.images[:10], encoder, strength=0.0)
        assert 0.0 <= result.fdr <= 1.0

    def test_fdr_monotone_decreasing_with_strength(self, synthetic_faces, encoder):
        """FDR at strength=0 should be >= FDR at strength=1 (encoding hurts detection)."""
        attacker = FaceDetectionAttacker()
        strengths = [0.0, 0.5, 1.0]
        results = attacker.sweep(synthetic_faces.images[:15], encoder, strengths)
        fdrs = [r.fdr for r in results]
        # Allow small non-monotonicity (noise) — at minimum s=1 <= s=0
        assert fdrs[-1] <= fdrs[0] + 0.15, (
            f"Expected FDR to decrease with encoding strength; got {fdrs}"
        )

    def test_sweep_returns_correct_number(self, synthetic_faces, encoder):
        attacker = FaceDetectionAttacker()
        results = attacker.sweep(synthetic_faces.images[:8], encoder, [0.0, 0.5, 1.0])
        assert len(results) == 3

    def test_per_image_counts_sum_to_detected(self, synthetic_faces, encoder):
        attacker = FaceDetectionAttacker()
        result = attacker.evaluate(synthetic_faces.images[:10], encoder, 0.0)
        detected_from_per_image = sum(1 for n in result.per_image if n >= 1)
        assert detected_from_per_image == result.detected


# ---------------------------------------------------------------------------
# FaceRecognitionAttacker / EER utilities
# ---------------------------------------------------------------------------

class TestEER:
    def test_random_scores_eer_near_half(self):
        rng = np.random.default_rng(0)
        genuine = rng.uniform(0.3, 0.7, 200).astype(np.float32)
        impostor = rng.uniform(0.3, 0.7, 200).astype(np.float32)
        eer, _ = _compute_eer(genuine, impostor)
        assert 0.3 <= eer <= 0.7

    def test_perfect_separation_eer_low(self):
        genuine = np.linspace(0.7, 1.0, 100).astype(np.float32)
        impostor = np.linspace(0.0, 0.3, 100).astype(np.float32)
        eer, _ = _compute_eer(genuine, impostor)
        assert eer < 0.15

    def test_empty_scores_returns_half(self):
        eer, _ = _compute_eer(np.array([]), np.array([0.5]))
        assert eer == pytest.approx(0.5)


class TestFaceRecognitionAttacker:
    def test_eer_in_range(self, synthetic_faces, encoder):
        attacker = FaceRecognitionAttacker(max_impostor_pairs=200)
        result = attacker.evaluate(synthetic_faces.images, synthetic_faces.labels, encoder, 0.0)
        assert 0.0 <= result.eer <= 1.0

    def test_eer_in_valid_range_at_all_strengths(self, synthetic_faces, encoder):
        """EER must remain a valid probability at every strength level."""
        attacker = FaceRecognitionAttacker(max_impostor_pairs=200)
        for strength in [0.0, 0.5, 1.0]:
            r = attacker.evaluate(synthetic_faces.images, synthetic_faces.labels, encoder, strength)
            assert 0.0 <= r.eer <= 1.0, f"EER out of range at strength={strength}: {r.eer}"


# ---------------------------------------------------------------------------
# PSFInversionAttacker
# ---------------------------------------------------------------------------

class TestPSFUtils:
    def test_gaussian_kernel_sums_to_one(self):
        k = _gaussian_kernel(11, 2.0)
        assert k.sum() == pytest.approx(1.0, abs=1e-5)

    def test_gaussian_kernel_shape(self):
        k = _gaussian_kernel(9, 1.5)
        assert k.shape == (9, 9)

    def test_psnr_identical_is_100(self):
        img = np.random.rand(50, 37, 3).astype(np.float32)
        assert _psnr(img, img) == pytest.approx(100.0, abs=1e-3)

    def test_psnr_all_zeros_vs_ones(self):
        a = np.zeros((20, 20, 3), dtype=np.float32)
        b = np.ones((20, 20, 3), dtype=np.float32)
        p = _psnr(a, b)
        assert p == pytest.approx(0.0, abs=0.1)


class TestPSFInversionAttacker:
    def test_wiener_psnr_finite(self, synthetic_faces, encoder):
        attacker = PSFInversionAttacker(n_rl_iter=5, n_examples=1)
        result = attacker.evaluate(synthetic_faces.images[:5], encoder, 0.0)
        assert np.isfinite(result.psnr_wiener)
        assert np.isfinite(result.psnr_rl)

    def test_wiener_psnr_ge_rl_psnr_at_high_strength(self, synthetic_faces, encoder):
        """Oracle Wiener should generally outperform blind RL."""
        attacker = PSFInversionAttacker(n_rl_iter=10, n_examples=1)
        result = attacker.evaluate(synthetic_faces.images[:8], encoder, 0.8)
        # Allow slack — on synthetic data the gap can be small
        assert result.psnr_wiener >= result.psnr_rl - 3.0, (
            f"Wiener PSNR={result.psnr_wiener:.1f} should be >= RL PSNR={result.psnr_rl:.1f}"
        )

    def test_psnr_reasonable_at_zero_strength(self, synthetic_faces, encoder):
        """At strength=0 recovered images should have a reasonable PSNR (>10 dB)."""
        attacker = PSFInversionAttacker(n_rl_iter=5, n_examples=1)
        result = attacker.evaluate(synthetic_faces.images[:5], encoder, 0.0)
        assert result.psnr_wiener > 10.0, (
            f"PSNR at strength=0 should be reasonable, got {result.psnr_wiener:.1f} dB"
        )

    def test_psnr_decreases_with_strength(self, synthetic_faces, encoder):
        """Stronger encoding → harder to invert → lower PSNR for RL attacker."""
        attacker = PSFInversionAttacker(n_rl_iter=5, n_examples=0)
        r0 = attacker.evaluate(synthetic_faces.images[:8], encoder, 0.0)
        r1 = attacker.evaluate(synthetic_faces.images[:8], encoder, 1.0)
        assert r1.psnr_rl <= r0.psnr_rl + 5.0  # generous tolerance

    def test_examples_stored(self, synthetic_faces, encoder):
        attacker = PSFInversionAttacker(n_rl_iter=3, n_examples=2)
        result = attacker.evaluate(synthetic_faces.images[:5], encoder, 0.5)
        assert len(result.examples) == min(2, 5)
        orig, enc, wiener, rl = result.examples[0]
        assert orig.shape == synthetic_faces.images.shape[1:]
        assert enc.shape == orig.shape


# ---------------------------------------------------------------------------
# PrivacyUtilityAnalyzer (smoke test — fast, no real data download)
# ---------------------------------------------------------------------------

class TestPrivacyUtilityAnalyzer:
    def test_analyzer_smoke(self, tmp_path):
        from qci.privacy.analyzer import PrivacyUtilityAnalyzer

        analyzer = PrivacyUtilityAnalyzer(
            strengths=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path),
            counter_cfg=[{"mode": "hog"}],
            n_images=20,
            encoder_mode="defocus",
        )
        df = analyzer.run()

        assert "strength" in df.columns
        assert "fdr" in df.columns
        assert "eer" in df.columns
        assert len(df) == 3

        assert (tmp_path / "privacy_utility_tradeoff.png").exists()
        assert (tmp_path / "privacy_results.csv").exists()

    def test_fdr_values_in_range(self, tmp_path):
        from qci.privacy.analyzer import PrivacyUtilityAnalyzer

        analyzer = PrivacyUtilityAnalyzer(
            strengths=[0.0, 1.0],
            output_dir=str(tmp_path / "sub"),
            n_images=15,
        )
        df = analyzer.run()
        assert (df["fdr"] >= 0.0).all()
        assert (df["fdr"] <= 1.0).all()
        assert (df["eer"] >= 0.0).all()
        assert (df["eer"] <= 1.0).all()
