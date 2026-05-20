"""
Ladder step 3a: learned 1st-order DS from one curved heee demonstration.

Trains PUMA on LASA heee demo 0, then simulates the learned field from a grid of
initial positions (compare with ladder2a straight-line demo).
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_MYSCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _MYSCRIPTS not in sys.path:
    sys.path.insert(0, _MYSCRIPTS)

from ladder3_common import run_learned_ds_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ladder 3a: learned 1st-order DS on one curved heee demo.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train; skip simulation plot.",
    )
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Load checkpoint and simulate (skip training).",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if a checkpoint exists.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Target iteration count; resumes from checkpoint if higher than last run.",
    )
    parser.add_argument("--steps", type=int, default=4000, help="Simulation steps.")
    parser.add_argument(
        "--n-simulate",
        type=int,
        default=None,
        help="During training, run ladder grid sim+plot every N iterations (omit to disable).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip interactive plot.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "images", "ladder3a_learned_ds.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_learned_ds_pipeline(
        "1st",
        train=not args.simulate_only,
        simulate=not args.train_only,
        force_train=args.force_train,
        max_iterations=args.max_iterations,
        n_simulate=args.n_simulate,
        n_steps=args.steps,
        live_plot=not args.no_plot,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
