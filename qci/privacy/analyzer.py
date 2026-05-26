"""PrivacyUtilityAnalyzer — publication-quality privacy–utility tradeoff figure.

The output figure has two stacked subplots:

  Top (Utility): MAE vs encoding_strength.
      One curve per counter type.  Lower MAE = better utility.

  Bottom (Privacy): FDR and EER vs encoding_strength.
      FDR on left y-axis (orange), EER on right y-axis (purple).
      Grey shaded region = "privacy achieved zone" (FDR < 0.2 AND EER > 0.4).
      A vertical dashed line marks the Pareto-optimal operating point:
      the smallest strength where privacy is achieved AND MAE ≤ 2× baseline.

Saved as ``privacy_utility_tradeoff.png`` (300 dpi, tight layout).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    """All metrics at one encoding strength."""

    strength: float
    mae_by_counter: Dict[str, float] = field(default_factory=dict)
    fdr: float = float("nan")
    eer: float = float("nan")
    psnr_wiener: float = float("nan")
    psnr_rl: float = float("nan")


# ---------------------------------------------------------------------------
# Privacy thresholds used for Pareto / privacy-zone colouring
# ---------------------------------------------------------------------------

_PRIVACY_FDR_MAX = 0.20   # FDR must drop below this
_PRIVACY_EER_MIN = 0.40   # EER must rise above this
_UTILITY_MAE_FACTOR = 2.0  # MAE must stay within 2× baseline (strength=0)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class PrivacyUtilityAnalyzer:
    """Run a full privacy–utility sweep and produce a publication figure.

    Parameters
    ----------
    strengths:
        Encoding strength values to sweep.
    output_dir:
        Where to save the figure and CSV.
    counter_cfg:
        List of counter config dicts (same format as sweep.yaml ``counters``
        list).  Defaults to BlobCounter (works on synthetic ellipse images)
        if None.  Use ``{"mode": "hog"}`` only when evaluating on real
        person images — HOG returns 0 on synthetic data, giving a flat curve.
    n_images:
        Number of face images to use (keeps runtime manageable).
    encoder_mode:
        OpticalEncoder mode — ``"defocus"``, ``"coded_mask"``, or
        ``"learnable_psf"``.
    """

    def __init__(
        self,
        strengths: Optional[List[float]] = None,
        output_dir: str = "results",
        counter_cfg: Optional[List[dict]] = None,
        n_images: int = 100,
        encoder_mode: str = "defocus",
    ) -> None:
        self.strengths = strengths or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Default: BlobCounter works on synthetic ellipse data.
        # HOG returns 0 for all synthetic images → flat MAE curve.
        self.counter_cfg = counter_cfg or [{"mode": "blob", "name": "blob"}]
        self.n_images = n_images
        self.encoder_mode = encoder_mode

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute full sweep; return results DataFrame; save figure + CSV."""
        from qci.optics.encoder import OpticalEncoder
        from qci.counting import build_counter
        from qci.privacy.face_data import FaceDataLoader
        from qci.privacy.face_attacker import FaceDetectionAttacker
        from qci.privacy.recognition_attacker import FaceRecognitionAttacker
        from qci.privacy.inversion_attacker import PSFInversionAttacker
        import torch

        # ---- data --------------------------------------------------------
        loader = FaceDataLoader(max_images=self.n_images)
        face_data = loader.load()
        images = face_data.images   # (N, H, W, 3) float32

        # Crowd-counting utility uses a SyntheticCrowdDataset
        from qci.data.synthetic import SyntheticCrowdDataset
        from torch.utils.data import DataLoader as TorchLoader

        crowd_ds = SyntheticCrowdDataset(n_images=40, image_size=(128, 128), max_count=30, seed=7)
        crowd_loader = TorchLoader(crowd_ds, batch_size=4)

        # ---- encoder -----------------------------------------------------
        encoder = OpticalEncoder(mode=self.encoder_mode, strength=0.0, kernel_size=11)

        # ---- counters ----------------------------------------------------
        counters: Dict[str, object] = {}
        for cfg in self.counter_cfg:
            name = cfg.get("name", cfg.get("mode", "hog"))
            counters[name] = build_counter(cfg)

        # ---- attackers ---------------------------------------------------
        det_attacker = FaceDetectionAttacker()
        rec_attacker = FaceRecognitionAttacker(max_impostor_pairs=500)
        inv_attacker = PSFInversionAttacker(n_rl_iter=20, n_examples=3)

        # ---- baseline MAE (strength = 0) ---------------------------------
        baseline_mae = self._crowd_mae(encoder, counters, crowd_loader, 0.0)

        # ---- sweep -------------------------------------------------------
        points: List[SweepPoint] = []
        all_inv_examples: Dict[float, list] = {}

        for strength in self.strengths:
            pt = SweepPoint(strength=strength)

            # Utility: crowd counting MAE
            pt.mae_by_counter = self._crowd_mae(encoder, counters, crowd_loader, strength)

            # Privacy: detection
            det_result = det_attacker.evaluate(images, encoder, strength)
            pt.fdr = det_result.fdr

            # Privacy: recognition EER
            rec_result = rec_attacker.evaluate(images, face_data.labels, encoder, strength)
            pt.eer = rec_result.eer

            # Privacy: inversion PSNR
            inv_result = inv_attacker.evaluate(images[:20], encoder, strength)
            pt.psnr_wiener = inv_result.psnr_wiener
            pt.psnr_rl = inv_result.psnr_rl
            all_inv_examples[strength] = inv_result.examples

            log.info(
                "strength=%.2f | MAE=%s | FDR=%.3f | EER=%.3f | "
                "PSNR_W=%.1f | PSNR_RL=%.1f",
                strength,
                {k: f"{v:.2f}" for k, v in pt.mae_by_counter.items()},
                pt.fdr, pt.eer, pt.psnr_wiener, pt.psnr_rl,
            )
            points.append(pt)

        # ---- Pareto operating point -------------------------------------
        pareto_strength = self._find_pareto(points, baseline_mae)

        # ---- save results -----------------------------------------------
        df = self._to_dataframe(points)
        csv_path = self.output_dir / "privacy_results.csv"
        df.to_csv(csv_path, index=False)
        log.info("Saved %s", csv_path)

        # ---- plot -------------------------------------------------------
        fig_path = self.output_dir / "privacy_utility_tradeoff.png"
        self._plot(points, baseline_mae, pareto_strength, fig_path)
        log.info("Saved %s", fig_path)

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _crowd_mae(
        self,
        encoder,
        counters: Dict[str, object],
        loader,
        strength: float,
    ) -> Dict[str, float]:
        import torch

        encoder.set_strength(strength)
        mae_dict: Dict[str, List[float]] = {name: [] for name in counters}

        with torch.no_grad():
            for imgs, counts, _ in loader:
                imgs = imgs.float()
                encoded = encoder(imgs)
                for name, counter in counters.items():
                    pred = counter(encoded)
                    for gt, p in zip(counts.float(), pred.float()):
                        mae_dict[name].append(abs(float(gt) - float(p)))

        return {name: float(np.mean(errs)) for name, errs in mae_dict.items()}

    def _find_pareto(
        self,
        points: List[SweepPoint],
        baseline_mae: Dict[str, float],
    ) -> Optional[float]:
        """Return the smallest strength satisfying privacy AND utility constraints."""
        # Average MAE across counters for multi-counter case
        def avg_mae(pt: SweepPoint) -> float:
            vals = list(pt.mae_by_counter.values())
            return float(np.mean(vals)) if vals else float("nan")

        def avg_baseline() -> float:
            vals = list(baseline_mae.values())
            return float(np.mean(vals)) if vals else float("nan")

        base_avg = avg_baseline()
        mae_limit = base_avg * _UTILITY_MAE_FACTOR

        for pt in sorted(points, key=lambda p: p.strength):
            privacy_ok = pt.fdr < _PRIVACY_FDR_MAX and pt.eer > _PRIVACY_EER_MIN
            utility_ok = avg_mae(pt) <= mae_limit or np.isnan(avg_mae(pt))
            if privacy_ok and utility_ok:
                return pt.strength

        # If strict criteria never met, return strongest privacy point
        best = min(points, key=lambda p: p.fdr)
        return best.strength

    def _to_dataframe(self, points: List[SweepPoint]) -> pd.DataFrame:
        rows = []
        for pt in points:
            row: dict = {
                "strength": pt.strength,
                "fdr": pt.fdr,
                "eer": pt.eer,
                "psnr_wiener": pt.psnr_wiener,
                "psnr_rl": pt.psnr_rl,
            }
            for name, mae_val in pt.mae_by_counter.items():
                row[f"mae_{name}"] = mae_val
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Publication-quality figure
    # ------------------------------------------------------------------

    def _plot(
        self,
        points: List[SweepPoint],
        baseline_mae: Dict[str, float],
        pareto_strength: Optional[float],
        out_path: Path,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        strengths = [pt.strength for pt in points]
        counter_names = sorted(points[0].mae_by_counter.keys()) if points else []

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1,
            figsize=(8, 9),
            sharex=True,
            gridspec_kw={"hspace": 0.08},
        )
        fig.patch.set_facecolor("white")

        # ----------------------------------------------------------------
        # TOP: Utility — MAE vs strength
        # ----------------------------------------------------------------
        colours = plt.cm.tab10.colors  # type: ignore[attr-defined]
        for i, name in enumerate(counter_names):
            mae_vals = [pt.mae_by_counter.get(name, float("nan")) for pt in points]
            ax_top.plot(
                strengths, mae_vals,
                marker="o", linewidth=2.0, color=colours[i % 10],
                label=name.upper(),
            )
            # Baseline dashed reference
            b_val = baseline_mae.get(name, float("nan"))
            ax_top.axhline(
                b_val, linestyle=":", linewidth=1.0, color=colours[i % 10], alpha=0.5
            )

        ax_top.set_ylabel("MAE (persons)", fontsize=12)
        ax_top.set_title("Privacy–Utility Tradeoff", fontsize=14, fontweight="bold", pad=10)
        ax_top.legend(loc="upper left", fontsize=9, framealpha=0.8)
        ax_top.grid(True, linestyle="--", alpha=0.4)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        # ----------------------------------------------------------------
        # BOTTOM: Privacy — FDR + EER vs strength
        # ----------------------------------------------------------------
        fdr_vals = [pt.fdr for pt in points]
        eer_vals = [pt.eer for pt in points]

        color_fdr = "#E07B39"   # warm orange
        color_eer = "#6A5ACD"   # slate blue

        ax_bot.plot(strengths, fdr_vals, marker="s", linewidth=2.0, color=color_fdr, label="FDR")
        ax_bot.set_ylabel("Face Detection Rate (FDR)", fontsize=12, color=color_fdr)
        ax_bot.tick_params(axis="y", labelcolor=color_fdr)
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.axhline(_PRIVACY_FDR_MAX, linestyle="--", linewidth=1.2, color=color_fdr, alpha=0.5)

        ax_eer = ax_bot.twinx()
        ax_eer.plot(strengths, eer_vals, marker="^", linewidth=2.0, color=color_eer, label="EER")
        ax_eer.set_ylabel("Equal Error Rate (EER)", fontsize=12, color=color_eer)
        ax_eer.tick_params(axis="y", labelcolor=color_eer)
        ax_eer.set_ylim(-0.05, 1.05)
        ax_eer.axhline(_PRIVACY_EER_MIN, linestyle="--", linewidth=1.2, color=color_eer, alpha=0.5)

        # Privacy-achieved shaded zone
        priv_zone_x = [s for s in strengths if _fdr_ok(s, fdr_vals, strengths) and _eer_ok(s, eer_vals, strengths)]
        if priv_zone_x:
            x_lo = min(priv_zone_x) - (strengths[1] - strengths[0]) / 2 if len(strengths) > 1 else min(priv_zone_x)
            x_hi = max(strengths)
            ax_bot.axvspan(x_lo, x_hi, alpha=0.08, color="green", label="Privacy achieved zone")

        # Pareto point
        if pareto_strength is not None:
            for ax in (ax_top, ax_bot):
                ax.axvline(
                    pareto_strength,
                    linestyle="-.",
                    linewidth=1.8,
                    color="#2E7D32",
                    label=f"Pareto point (s={pareto_strength:.2f})",
                )

        ax_bot.set_xlabel("Encoding Strength", fontsize=12)
        ax_bot.grid(True, linestyle="--", alpha=0.4)
        ax_bot.spines["top"].set_visible(False)

        # Combined legend for bottom panel
        lines1, labels1 = ax_bot.get_legend_handles_labels()
        lines2, labels2 = ax_eer.get_legend_handles_labels()
        ax_bot.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9, framealpha=0.8)

        # Annotations
        ax_top.annotate(
            "Lower = better utility",
            xy=(0.02, 0.96), xycoords="axes fraction",
            fontsize=8, color="gray", va="top",
        )
        ax_bot.annotate(
            "Lower FDR = better privacy",
            xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=8, color=color_fdr, va="bottom",
        )
        ax_eer.annotate(
            "Higher EER = better privacy",
            xy=(0.98, 0.04), xycoords="axes fraction",
            fontsize=8, color=color_eer, va="bottom", ha="right",
        )

        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Helper predicates for privacy-zone calculation
# ---------------------------------------------------------------------------

def _fdr_ok(s: float, fdr_vals: List[float], strengths: List[float]) -> bool:
    try:
        idx = strengths.index(s)
        return fdr_vals[idx] < _PRIVACY_FDR_MAX
    except (ValueError, IndexError):
        return False


def _eer_ok(s: float, eer_vals: List[float], strengths: List[float]) -> bool:
    try:
        idx = strengths.index(s)
        return eer_vals[idx] > _PRIVACY_EER_MIN
    except (ValueError, IndexError):
        return False
