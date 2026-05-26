#!/usr/bin/env python3
"""run_privacy_eval.py — full privacy sweep producing tradeoff figure and CSV.

Outputs (all in --output_dir):
  privacy_utility_tradeoff.png   — publication-quality 2-panel figure
  privacy_results.csv            — per-strength metrics table
  inversion_examples/            — grid images showing original vs encoded vs recovered

Usage::

    python scripts/run_privacy_eval.py
    python scripts/run_privacy_eval.py --mode coded_mask --strengths 0 0.3 0.6 1.0
    python scripts/run_privacy_eval.py --n_images 200 --output_dir results/privacy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_privacy_eval")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Privacy evaluation sweep.")
    p.add_argument("--mode", default="defocus",
                   choices=["defocus", "coded_mask", "learnable_psf"],
                   help="Optical encoder mode.")
    p.add_argument("--strengths", nargs="+", type=float,
                   default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   help="Encoding strengths to sweep.")
    p.add_argument("--n_images", type=int, default=100,
                   help="Number of face images to use (caps LFW load).")
    p.add_argument("--n_rl_iter", type=int, default=20,
                   help="Richardson–Lucy iterations.")
    p.add_argument("--output_dir", default="results",
                   help="Directory for outputs.")
    p.add_argument("--counters", nargs="+", default=["hog"],
                   choices=["hog", "csrnet", "density"],
                   help="Counters for utility evaluation.")
    return p


def _save_inversion_grid(
    examples_by_strength: dict,
    output_dir: Path,
) -> None:
    """Save a 4-column grid (original / encoded / Wiener / RL) for each strength."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        inv_dir = output_dir / "inversion_examples"
        inv_dir.mkdir(exist_ok=True)

        for strength, examples in examples_by_strength.items():
            if not examples:
                continue
            n_rows = len(examples)
            fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows), squeeze=False)
            col_labels = ["Original", "Encoded", "Wiener (oracle)", "RL (blind)"]

            for row_idx, (orig, enc, wiener, rl) in enumerate(examples):
                for col_idx, (img, lbl) in enumerate(
                    zip([orig, enc, wiener, rl], col_labels)
                ):
                    ax = axes[row_idx][col_idx]
                    ax.imshow(np.clip(img, 0, 1))
                    ax.axis("off")
                    if row_idx == 0:
                        ax.set_title(lbl, fontsize=11, fontweight="bold")

            fig.suptitle(f"PSF Inversion — strength={strength:.2f}", fontsize=13, y=1.01)
            fig.tight_layout()
            fname = inv_dir / f"inversion_s{strength:.2f}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved %s", fname)
    except Exception as exc:
        log.warning("Could not save inversion grids: %s", exc)


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from qci.privacy.analyzer import PrivacyUtilityAnalyzer
    from qci.privacy.face_data import FaceDataLoader
    from qci.privacy.inversion_attacker import PSFInversionAttacker
    from qci.optics.encoder import OpticalEncoder
    import torch

    counter_cfg = [{"mode": c} for c in args.counters]

    analyzer = PrivacyUtilityAnalyzer(
        strengths=args.strengths,
        output_dir=str(output_dir),
        counter_cfg=counter_cfg,
        n_images=args.n_images,
        encoder_mode=args.mode,
    )

    df = analyzer.run()
    print("\n=== Privacy Evaluation Results ===")
    print(df.to_string(index=False, float_format="{:.3f}".format))
    print(f"\nFigure saved: {output_dir}/privacy_utility_tradeoff.png")
    print(f"CSV   saved: {output_dir}/privacy_results.csv")

    # Also produce inversion example grids separately
    log.info("Generating inversion example grids...")
    encoder = OpticalEncoder(mode=args.mode, strength=0.0, kernel_size=11)
    loader = FaceDataLoader(max_images=30)
    face_data = loader.load()
    inv_attacker = PSFInversionAttacker(n_rl_iter=args.n_rl_iter, n_examples=3)

    examples_by_strength = {}
    for s in args.strengths:
        result = inv_attacker.evaluate(face_data.images[:20], encoder, s)
        examples_by_strength[s] = result.examples

    _save_inversion_grid(examples_by_strength, output_dir)


if __name__ == "__main__":
    main()
