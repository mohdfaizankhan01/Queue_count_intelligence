#!/usr/bin/env python3
"""Run the encoding-strength sweep experiment.

Single-counter mode (L2)::

    python scripts/run_sweep.py --config configs/sweep.yaml

Multi-counter mode (L3)::

    python scripts/run_sweep.py --config configs/sweep.yaml --all-counters

The multi-counter sweep evaluates HOG, YOLO, and CSRNet on the *same*
encoded measurements and produces ``results/sweep_counters.png`` — one
subplot per crowd regime, one curve per counter.

Optional geometry projection (L4)::

    python scripts/run_sweep.py --config configs/sweep.yaml --geometry

Override any config key inline::

    python scripts/run_sweep.py --config configs/sweep.yaml \\
        --set encoding.mode=coded_mask \\
        --set sweep.ablate_recovery=true
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qci.eval.runner import run_multicounter_sweep, run_sweep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for override in overrides:
        key_path, _, raw_val = override.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            node[keys[-1]] = yaml.safe_load(raw_val)
        except yaml.YAMLError:
            node[keys[-1]] = raw_val
    return cfg


def _print_summary(df, group_cols: list[str]) -> None:
    cols = [c for c in group_cols if c in df.columns]
    agg = df.groupby(cols).agg(mae=("ae", "mean"), n=("ae", "count")).reset_index()
    print("\n=== Sweep results (MAE per condition) ===")
    print(agg.to_string(index=False, float_format="{:.2f}".format))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QCI encoding-strength sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        dest="overrides",
        help="Override a config key, e.g. --set encoding.mode=defocus",
    )
    parser.add_argument(
        "--all-counters",
        action="store_true",
        default=False,
        help="Run HOG, YOLO, and CSRNet counters and produce sweep_counters.png",
    )
    parser.add_argument(
        "--geometry",
        action="store_true",
        default=False,
        help="Project detections to ground plane before counting (L4)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg = _apply_overrides(cfg, args.overrides)
    logging.getLogger("qci").setLevel(logging.INFO)

    if args.geometry:
        cfg.setdefault("geometry", {})["enabled"] = True

    if args.all_counters:
        # Inject default counter list if not already in config
        if "counters" not in cfg:
            cfg["counters"] = [
                {"name": "hog",     "mode": "hog"},
                {"name": "yolo",    "mode": "yolo"},
                {"name": "csrnet",  "mode": "csrnet", "pretrained_frontend": True},
            ]
        df = run_multicounter_sweep(cfg)
        _print_summary(df, ["strength", "counter", "crowd_regime"])
        out = cfg.get("output", {})
        print(f"\nFull results: {out.get('counters_csv', 'results/counters_results.csv')}")
        print(f"Plot:         {out.get('counters_plot', 'results/sweep_counters.png')}")
    else:
        df = run_sweep(cfg)
        _print_summary(df, ["strength", "recovery"])
        out = cfg.get("output", {})
        print(f"\nFull results: {out.get('results_csv', 'results/results.csv')}")
        print(f"Plot:         {out.get('plot_path', 'results/sweep_plot.png')}")


if __name__ == "__main__":
    main()
