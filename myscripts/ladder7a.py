"""
Ladder step 7a: 1st-order DS transported under non-uniform scaling.

Pulls a source field from any prior ladder step (analytic straight-line or learned
checkpoints from steps 2–4), applies a non-uniform scale transform, and simulates in the
target frame.
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

from ladder7_common import add_transport_cli_args, run_transport_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ladder 7a: transported 1st-order DS (non-uniform scaling).",
    )
    add_transport_cli_args(parser)
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "images", "ladder7a_transported_ds.png"),
        help="Path for output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np

    run_transport_pipeline(
        "1st",
        source=args.source,
        scale=np.array(args.scale, dtype=float),
        translation=np.array(args.translation, dtype=float),
        attractor=np.array(args.attractor, dtype=float),
        gain=args.gain,
        dt=args.dt,
        delta_t=args.delta_t,
        n_steps=args.steps,
        live_plot=not args.no_plot,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
