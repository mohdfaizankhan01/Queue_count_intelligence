#!/usr/bin/env python3
"""run_full_pipeline.py — master script: runs all layers, produces summary_table.csv.

Execution order
---------------
1. Crowd-counting sweep (Layers 1–3)   → results/counters_results.csv
2. Queue analytics (Layer 5)            → results/history.db  (demo insert)
3. Privacy evaluation (Layer 6)         → results/privacy_results.csv
                                          results/privacy_utility_tradeoff.png
4. Assemble summary_table.csv from the above outputs.

Usage::

    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --config configs/sweep.yaml --output_dir results
    python scripts/run_full_pipeline.py --skip_sweep   # skip slow counter sweep
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full QCI pipeline runner.")
    p.add_argument("--config", default="configs/sweep.yaml", help="Sweep config YAML.")
    p.add_argument("--output_dir", default="results", help="Output directory.")
    p.add_argument("--skip_sweep", action="store_true", help="Skip the counter sweep.")
    p.add_argument("--skip_privacy", action="store_true", help="Skip privacy evaluation.")
    p.add_argument("--n_privacy_images", type=int, default=60, help="Face images for privacy eval.")
    p.add_argument("--privacy_strengths", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="Encoding strengths for privacy sweep.")
    p.add_argument("--encoder_mode", default="defocus",
                   choices=["defocus", "coded_mask", "learnable_psf"],
                   help="Optical encoder mode for privacy eval.")
    return p


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_sweep(config_path: str, output_dir: Path) -> None:
    log.info("=== STAGE 1: Crowd-counting sweep ===")
    import yaml
    from qci.eval.runner import run_multicounter_sweep

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["output_dir"] = str(output_dir)
    run_multicounter_sweep(cfg)
    log.info("Counter sweep done → %s/counters_results.csv", output_dir)


def _run_analytics_demo(output_dir: Path) -> None:
    log.info("=== STAGE 2: Queue analytics demo insert ===")
    from qci.analytics import ServiceRateModel, QueueStatus, HistoryTracker

    db_path = output_dir / "history.db"
    tracker = HistoryTracker(db_path=str(db_path))
    model = ServiceRateModel(n_booths=3, avg_service_time_sec=120)

    demo_stations = [
        ("STATION_A", 47, 23.0, 0.8),
        ("STATION_B", 12, 8.5, 0.4),
        ("STATION_C", 85, 41.0, 1.2),
    ]
    for sid, count, q_len, density in demo_stations:
        wait = model.estimate_wait(count)
        status = QueueStatus.create(
            station_id=sid,
            person_count=count,
            queue_length_m=q_len,
            crowd_density=density,
            wait_estimate=wait,
        )
        tracker.insert(status)
        log.info("  %s", status.summary())

    tracker.close()
    log.info("Analytics demo done → %s", db_path)


def _run_privacy(
    output_dir: Path,
    n_images: int,
    strengths: list,
    encoder_mode: str,
) -> None:
    log.info("=== STAGE 3: Privacy evaluation sweep ===")
    from qci.privacy.analyzer import PrivacyUtilityAnalyzer

    analyzer = PrivacyUtilityAnalyzer(
        strengths=strengths,
        output_dir=str(output_dir),
        counter_cfg=[{"mode": "hog"}],
        n_images=n_images,
        encoder_mode=encoder_mode,
    )
    df = analyzer.run()
    log.info("Privacy eval done → %s/privacy_results.csv", output_dir)
    return df


def _build_summary(output_dir: Path) -> None:
    log.info("=== STAGE 4: Building summary_table.csv ===")
    import pandas as pd

    rows = []

    # Counter sweep results
    sweep_csv = output_dir / "counters_results.csv"
    if sweep_csv.exists():
        sweep_df = pd.read_csv(sweep_csv)
        if not sweep_df.empty:
            # Summarise: mean MAE per (counter, strength)
            grp_cols = [c for c in ("counter", "strength") if c in sweep_df.columns]
            if grp_cols and "abs_error" in sweep_df.columns:
                summary = sweep_df.groupby(grp_cols)["abs_error"].mean().reset_index()
                summary.rename(columns={"abs_error": "mean_mae"}, inplace=True)
                summary["source"] = "counter_sweep"
                rows.append(summary)

    # Privacy results
    priv_csv = output_dir / "privacy_results.csv"
    if priv_csv.exists():
        priv_df = pd.read_csv(priv_csv)
        priv_df["source"] = "privacy_eval"
        rows.append(priv_df)

    import pandas as _pd
    if rows:
        summary_df = _pd.concat(rows, ignore_index=True, sort=False)
    else:
        summary_df = _pd.DataFrame({"note": ["No results found — run without --skip flags."]})

    out_path = output_dir / "summary_table.csv"
    summary_df.to_csv(out_path, index=False)
    log.info("Summary saved → %s", out_path)
    print("\n=== SUMMARY TABLE (first 20 rows) ===")
    print(summary_df.head(20).to_string(index=False))


# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_sweep:
        try:
            _run_sweep(args.config, output_dir)
        except Exception as exc:
            log.warning("Counter sweep failed (continuing): %s", exc)

    _run_analytics_demo(output_dir)

    if not args.skip_privacy:
        try:
            _run_privacy(
                output_dir,
                n_images=args.n_privacy_images,
                strengths=args.privacy_strengths,
                encoder_mode=args.encoder_mode,
            )
        except Exception as exc:
            log.warning("Privacy eval failed (continuing): %s", exc)

    _build_summary(output_dir)
    log.info("Pipeline complete.  Outputs in: %s/", output_dir)


if __name__ == "__main__":
    main()
