"""Headline sweep experiment: counting error as a function of encoding strength.

Forward model:  x  →  encode(strength)  →  degrade  →  recover  →  count  →  error

Layer 3 extensions
------------------
* ``crowd_regime`` column in every record (based on GT count).
* ``run_multicounter_sweep`` runs HOG / YOLO / CSRNet on the *same* measurements
  for a fair per-strength comparison.

.. TODO Layer 6: plug privacy attacker on measurement y between encode and count.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qci.counting import build_counter, crowd_regime
from qci.data import build_dataset
from qci.eval.metrics import absolute_error, aggregate_metrics, game, squared_error
from qci.optics.degradation import build_degradation
from qci.optics.encoder import build_encoder
from qci.recovery import build_restoration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-condition pass (one counter, one strength)
# ---------------------------------------------------------------------------


def _run_condition(
    dataset,
    encoder,
    degradation,
    recovery,
    counter,
    strength: float,
    recovery_label: str,
    device: torch.device,
    counter_name: str = "counter",
    batch_size: int = 1,
) -> list[dict]:
    """Run one (strength, recovery, counter) combination; return per-image records."""
    encoder.set_strength(strength)
    encoder.eval()
    recovery.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    records = []

    with torch.no_grad():
        for images, gt_counts, gt_dmaps in loader:
            images = images.to(device)

            # --- encode ---
            y = encoder(images)

            # --- degrade (optional) ---
            if degradation is not None:
                y = degradation(y)

            # --- recover ---
            y_rec = recovery(y)

            # --- count ---
            pred_counts = counter(y_rec.cpu())

            # --- record ---
            for i in range(images.shape[0]):
                pred = float(pred_counts[i])
                gt = float(gt_counts[i])

                pred_dmap = gt_dmaps[i, 0].numpy()
                gt_sum = float(gt_dmaps[i].sum())
                if gt_sum > 0:
                    pred_dmap_scaled = pred_dmap * (pred / (gt_sum + 1e-8))
                else:
                    pred_dmap_scaled = pred_dmap * 0.0
                gt_dmap_np = gt_dmaps[i, 0].numpy()

                rec: dict = {
                    "strength": strength,
                    "recovery": recovery_label,
                    "counter": counter_name,
                    "pred": pred,
                    "gt": float(gt),
                    "ae": absolute_error(pred, float(gt)),
                    "se": squared_error(pred, float(gt)),
                    "crowd_regime": crowd_regime(float(gt)),
                }
                for lvl in range(4):
                    rec[f"game_{lvl}"] = game(pred_dmap_scaled, gt_dmap_np, level=lvl)
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Multi-counter pass: encode once, run N counters on the same measurements
# ---------------------------------------------------------------------------


def _run_multicounter_condition(
    dataset,
    encoder,
    degradation,
    recovery,
    counters: dict,          # {name: CrowdCounter}
    strength: float,
    device: torch.device,
    batch_size: int = 1,
) -> list[dict]:
    """Encode/degrade/recover once, then evaluate all counters on the same y_rec."""
    encoder.set_strength(strength)
    encoder.eval()
    recovery.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    records = []

    with torch.no_grad():
        for images, gt_counts, gt_dmaps in loader:
            images = images.to(device)

            y = encoder(images)
            if degradation is not None:
                y = degradation(y)
            y_rec = recovery(y)

            for cname, counter in counters.items():
                pred_counts = counter(y_rec.cpu())

                for i in range(images.shape[0]):
                    pred = float(pred_counts[i])
                    gt = float(gt_counts[i])

                    pred_dmap = gt_dmaps[i, 0].numpy()
                    gt_sum = float(gt_dmaps[i].sum())
                    pred_dmap_scaled = (
                        pred_dmap * (pred / (gt_sum + 1e-8)) if gt_sum > 0 else pred_dmap * 0.0
                    )

                    rec: dict = {
                        "strength": strength,
                        "counter": cname,
                        "pred": pred,
                        "gt": gt,
                        "ae": absolute_error(pred, gt),
                        "se": squared_error(pred, gt),
                        "crowd_regime": crowd_regime(gt),
                    }
                    for lvl in range(4):
                        rec[f"game_{lvl}"] = game(pred_dmap_scaled, gt_dmaps[i, 0].numpy(), level=lvl)
                    records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Single-counter sweep runner (L2 API, backward compatible)
# ---------------------------------------------------------------------------


class SweepRunner:
    """Orchestrates the single-counter encoding-strength sweep."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu"
        )
        logger.info(f"Sweep device: {self.device}")

    def run(self) -> pd.DataFrame:
        cfg = self.cfg
        dataset = build_dataset(cfg["dataset"])
        encoder = build_encoder(cfg["encoding"]).to(self.device)
        degradation = build_degradation(cfg.get("degradation", {"enabled": False}))
        if degradation is not None:
            degradation = degradation.to(self.device)

        counter_cfg = cfg.get("counting", {"mode": "hog"})
        strengths: list[float] = cfg["sweep"]["strengths"]
        ablate: bool = cfg["sweep"].get("ablate_recovery", False)
        batch_size: int = cfg.get("batch_size", 1)
        all_records: list[dict] = []

        for strength in tqdm(strengths, desc="Encoding strength sweep"):
            recovery_cfg = cfg.get("recovery", {"mode": "none"})
            recovery = build_restoration(recovery_cfg)
            if recovery_cfg.get("mode") == "wiener":
                recovery.set_psf(encoder.get_psf_tensor())

            counter = build_counter(counter_cfg)
            counter_name = counter_cfg.get("mode", "counter")

            records = _run_condition(
                dataset, encoder, degradation, recovery, counter,
                strength, "recovery", self.device, counter_name, batch_size,
            )
            all_records.extend(records)

            if ablate:
                no_recovery = build_restoration({"mode": "none"})
                records_no = _run_condition(
                    dataset, encoder, degradation, no_recovery, counter,
                    strength, "no_recovery", self.device, counter_name, batch_size,
                )
                all_records.extend(records_no)

        return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Multi-counter sweep runner (L3)
# ---------------------------------------------------------------------------


class MultiCounterSweepRunner:
    """Sweep with multiple counters evaluated on identical measurements."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu"
        )

    def run(self) -> pd.DataFrame:
        cfg = self.cfg
        dataset = build_dataset(cfg["dataset"])
        encoder = build_encoder(cfg["encoding"]).to(self.device)
        degradation = build_degradation(cfg.get("degradation", {"enabled": False}))
        if degradation is not None:
            degradation = degradation.to(self.device)

        # Build all requested counters
        counter_cfgs: list[dict] = cfg.get("counters", [{"mode": "hog"}])
        counters: dict = {}
        for ccfg in counter_cfgs:
            name = ccfg.get("name", ccfg.get("mode", "counter"))
            counters[name] = build_counter(ccfg)

        recovery_cfg = cfg.get("recovery", {"mode": "none"})
        strengths: list[float] = cfg["sweep"]["strengths"]
        batch_size: int = cfg.get("batch_size", 1)
        all_records: list[dict] = []

        for strength in tqdm(strengths, desc="Multi-counter sweep"):
            recovery = build_restoration(recovery_cfg)
            if recovery_cfg.get("mode") == "wiener":
                recovery.set_psf(encoder.get_psf_tensor())

            records = _run_multicounter_condition(
                dataset, encoder, degradation, recovery, counters,
                strength, self.device, batch_size,
            )
            all_records.extend(records)

        return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def run_sweep(cfg: dict) -> pd.DataFrame:
    """Single-counter sweep (L2 API, backward compatible).  Saves CSV + plot."""
    import matplotlib
    matplotlib.use("Agg")

    runner = SweepRunner(cfg)
    df = runner.run()

    out_cfg = cfg.get("output", {})
    csv_path = Path(out_cfg.get("results_csv", "results/results.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    agg = (
        df.groupby(["strength", "recovery"])
        .agg(mae=("ae", "mean"), rmse=("se", lambda s: np.sqrt(s.mean())))
        .reset_index()
    )
    _plot_sweep(agg, out_cfg.get("plot_path", "results/sweep_plot.png"))
    return df


def run_multicounter_sweep(cfg: dict) -> pd.DataFrame:
    """Multi-counter sweep (L3).  Saves CSV + regime-overlay plot."""
    import matplotlib
    matplotlib.use("Agg")

    runner = MultiCounterSweepRunner(cfg)
    df = runner.run()

    out_cfg = cfg.get("output", {})
    csv_path = Path(out_cfg.get("counters_csv", "results/counters_results.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Multi-counter results saved to {csv_path}")

    plot_path = out_cfg.get("counters_plot", "results/sweep_counters.png")
    _plot_counters_by_regime(df, plot_path)
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_sweep(agg: pd.DataFrame, plot_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, grp in agg.groupby("recovery"):
        ax.plot(grp["strength"], grp["mae"], marker="o", label=label)
        ax.fill_between(
            grp["strength"],
            (grp["mae"] - grp["rmse"]).clip(lower=0),
            grp["mae"] + grp["rmse"],
            alpha=0.15,
        )
    ax.set_xlabel("Encoding strength", fontsize=13)
    ax.set_ylabel("MAE (count)", fontsize=13)
    ax.set_title("Counting error vs. optical encoding strength", fontsize=14)
    ax.legend(title="Recovery")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot saved to {plot_path}")


def _plot_counters_by_regime(df: pd.DataFrame, plot_path: str) -> None:
    """MAE vs strength, one subplot per crowd regime, one curve per counter."""
    import matplotlib.pyplot as plt
    from qci.counting import REGIMES

    regimes_present = [r for r in REGIMES if r in df["crowd_regime"].unique()]
    n = len(regimes_present)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    counters = sorted(df["counter"].unique())
    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax, regime in zip(axes, regimes_present):
        sub = df[df["crowd_regime"] == regime]
        if sub.empty:
            ax.set_title(f"{regime.capitalize()} (no data)")
            continue

        agg = (
            sub.groupby(["strength", "counter"])
            .agg(mae=("ae", "mean"), rmse=("se", lambda s: np.sqrt(s.mean())))
            .reset_index()
        )
        for idx, cname in enumerate(counters):
            cgrp = agg[agg["counter"] == cname]
            if cgrp.empty:
                continue
            col = colours[idx % len(colours)]
            ax.plot(cgrp["strength"], cgrp["mae"], marker="o", label=cname, color=col)
            ax.fill_between(
                cgrp["strength"],
                (cgrp["mae"] - cgrp["rmse"]).clip(lower=0),
                cgrp["mae"] + cgrp["rmse"],
                alpha=0.12,
                color=col,
            )

        ax.set_title(f"{regime.capitalize()} crowd", fontsize=12)
        ax.set_xlabel("Encoding strength", fontsize=11)
        ax.set_ylabel("MAE (count)", fontsize=11)
        ax.legend(title="Counter", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("MAE vs Encoding Strength — by Crowd Regime & Counter", fontsize=13)
    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Counter-regime plot saved to {plot_path}")
