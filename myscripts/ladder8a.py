"""
Ladder step 8a: 1st-order keypoint policy transport.

Pulls a source field from any prior ladder step (analytic straight-line or learned
checkpoints from steps 2–4), transports via dy/dt = J_phi(x) f(x) with phi = psi o A
(affine map + nonlinear shape deformation), and simulates in the target frame.
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

from ladder8_common import add_policy_transport_cli_args, run_policy_transport_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ladder 8a: policy-transported 1st-order DS (affine + nonlinear deform).",
    )
    add_policy_transport_cli_args(parser)
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "images",
            "ladder8a_policy_transport_ds.png",
        ),
        help="Path for output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np

    run_policy_transport_pipeline(
        "1st",
        source=args.source,
        angle_rad=np.deg2rad(args.rotation_deg),
        scale=np.array(args.scale, dtype=float),
        translation=np.array(args.translation, dtype=float),
        deform_amplitude=args.deform_amplitude,
        deform_omega=np.array(args.deform_freq, dtype=float),
        deform_seed=args.deform_seed,
        attractor=np.array(args.attractor, dtype=float),
        gain=args.gain,
        dt=args.dt,
        delta_t=args.delta_t,
        n_steps=args.steps,
        live_plot=not args.no_plot,
        save_path=args.save,
        random_map=args.random_map,
        map_seed=args.map_seed,
    )


if __name__ == "__main__":
    main()
