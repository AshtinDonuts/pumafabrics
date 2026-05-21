"""
Ladder step 11a: 1st-order DS transported from source curve to a real demo target curve.

Fits a PolicyTransportation2D map (affine Procrustes + Gaussian-RBF residual) directly
from the source system's demonstration keypoints to a target curve drawn from the LASA
or LAIR dataset.  The transported field is then simulated from a grid of initial states.

Usage examples
--------------
# Analytic source → LASA heee target (default)
python3 myscripts/ladder11a.py --source analytic --no-plot

# Analytic source → LASA CShape
python3 myscripts/ladder11a.py --source analytic --target-dataset lasa --target-name CShape --no-plot

# Analytic source → LAIR capricorn
python3 myscripts/ladder11a.py --source analytic --target-dataset lair --target-name capricorn --no-plot

# Learned source (train ladder 3 first) → LASA Sine
python3 myscripts/ladder11a.py --source learned3 --target-dataset lasa --target-name Sine --no-plot

# Learned source → LAIR mountain, more keypoints, custom bandwidth
python3 myscripts/ladder11a.py --source learned4 --target-dataset lair --target-name mountain \\
    --n-keypoints 60 --rbf-bandwidth 0.2 --no-plot
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

from ladder11_common import add_demo_transport_cli_args, run_demo_transport_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ladder 11a: 1st-order DS transported to a real LASA/LAIR demo curve "
            "(affine Procrustes + Gaussian-RBF residual)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_demo_transport_cli_args(parser)
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "images",
            "ladder11a_demo_transport_ds.png",
        ),
        help="Path for output figure (default: myscripts/images/ladder11a_demo_transport_ds.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np

    run_demo_transport_pipeline(
        "1st",
        source=args.source,
        target_dataset=args.target_dataset,
        target_name=args.target_name,
        target_demo=args.target_demo,
        n_keypoints=args.n_keypoints,
        do_scale=not args.no_affine_scale,
        do_rotation=not args.no_affine_rotation,
        rbf_bandwidth=args.rbf_bandwidth,
        rbf_reg=args.rbf_reg,
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
        grid_margin=args.grid_margin,
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
