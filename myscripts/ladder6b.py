"""
Ladder step 6b: 2nd-order DS transported under uniform scaling.

Pulls a source field from any prior ladder step (analytic straight-line or learned
checkpoints from steps 2–4), applies a uniform scale transform, and simulates in the
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

from ladder6_common import add_transport_cli_args, run_transport_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ladder 6b: transported 2nd-order DS (uniform scaling).",
    )
    add_transport_cli_args(parser)
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "images",
            "ladder6b_transported_ds_2nd_order.png",
        ),
        help="Path for output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np

    run_transport_pipeline(
        "2nd",
        source=args.source,
        scale=args.scale,
        translation=np.array(args.translation, dtype=float),
        attractor=np.array(args.attractor, dtype=float),
        kp=args.kp,
        kd=args.kd,
        dt=args.dt,
        delta_t=args.delta_t,
        n_steps=args.steps,
        live_plot=not args.no_plot,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
