"""
Load and plot the analytic straight-line demonstration stored as a pickle
(`x_pos` layout compatible with `plot_3D.py`, same convention as `plot_lasa.py` export).

Examples:
  # Write the default pickle from ladder2_line straight-line .npy
  python myscripts/utils/plot_demo.py --mode export

  # Plot default myscripts/utils/analytical.pkl
  python myscripts/utils/plot_demo.py

  # Plot a custom pickle
  python myscripts/utils/plot_demo.py --pkl /tmp/demo.pk --savefig /tmp/demo.png
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_MYSCRIPTS = os.path.dirname(_UTILS_DIR)
_REPO_ROOT = os.path.abspath(os.path.join(_MYSCRIPTS, ".."))
_DEFAULT_PKL = os.path.join(_UTILS_DIR, "analytical.pkl")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _MYSCRIPTS not in sys.path:
    sys.path.insert(0, _MYSCRIPTS)


def _default_dt() -> float:
    return 0.03


def load_demo_pickle(pkl_path: str) -> dict:
    pkl_path = os.path.expanduser(pkl_path)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Not a file: {pkl_path}")
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in pickle, got {type(obj)}")
    return obj


def _trajectories_from_payload(obj: dict) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    Returns list of (sx, sy, label) per trajectory.

    Supports:
      - `x_pos` / `pos_fk`: single demonstration
      - `demos`: list of x_pos-like sequences
    """
    if "demos" in obj:
        out: list[tuple[np.ndarray, np.ndarray, str]] = []
        for i, xyz in enumerate(obj["demos"]):
            sx, sy = _xy_from_xyz(xyz)
            out.append((sx, sy, f"demo {i}"))
        return out

    xyz = None
    for key in ("x_pos", "pos_fk"):
        if key in obj:
            xyz = obj[key]
            break
    if xyz is None:
        keys = ", ".join(sorted(obj.keys()))
        raise KeyError(
            "Pickle has no trajectory data. Expected `x_pos`, `pos_fk`, or `demos`.\n"
            f"Top-level keys: {keys}"
        )
    sx, sy = _xy_from_xyz(xyz)
    return [(sx, sy, "demonstration")]


def _xy_from_xyz(xyz) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected (T, D>=2) trajectory, got shape {arr.shape}")
    return arr[:, 0], arr[:, 1]


def trajectory_to_plot3d_dict(
    pts_txd: np.ndarray,
    dt: float,
    source: str,
    attractor: np.ndarray | None = None,
) -> dict:
    """Build a dict compatible with `plot_3D.py` (`x_pos` as [x, y, 0])."""
    pts_txd = np.asarray(pts_txd, dtype=np.float64)
    if pts_txd.ndim != 2 or pts_txd.shape[1] < 2:
        raise ValueError(f"Expected (T, D>=2), got shape {pts_txd.shape}")
    t = pts_txd.shape[0]
    if pts_txd.shape[1] == 2:
        x_pos = [[float(pts_txd[i, 0]), float(pts_txd[i, 1]), 0.0] for i in range(t)]
    else:
        x_pos = [
            [float(pts_txd[i, 0]), float(pts_txd[i, 1]), float(pts_txd[i, 2])] for i in range(t)
        ]
    payload: dict = {
        "x_pos": x_pos,
        "delta_t": np.full(t, float(dt), dtype=np.float64),
        "source": source,
        "format": "analytical_export_for_plot_3D",
    }
    if attractor is not None:
        att = np.asarray(attractor, dtype=np.float64).reshape(-1)
        payload["attractor"] = att.tolist() if att.size > 2 else [float(att[0]), float(att[1]), 0.0]
    return payload


def export_analytical_pickle(out_path: str, dt: float | None = None) -> str:
    """Export the ladder2 straight-line demo to the default pickle layout."""
    from ladder2_data import demo_path, ensure_straight_line_dataset, load_demonstration_xy

    ensure_straight_line_dataset()
    pts = load_demonstration_xy()
    attractor = pts[-1]
    source = os.path.abspath(demo_path())
    payload = trajectory_to_plot3d_dict(
        pts,
        dt=_default_dt() if dt is None else dt,
        source=source,
        attractor=attractor,
    )
    out_path = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(
        f"Wrote {out_path} ({len(payload['x_pos'])} points, attractor={attractor.tolist()})",
        file=sys.stderr,
    )
    return out_path


def run_plot(
    pkl_path: str,
    title: str | None = None,
    savefig: str | None = None,
    show: bool = True,
) -> None:
    if plt is None:
        raise SystemExit("matplotlib is required for --mode plot (`pip install matplotlib`).")

    obj = load_demo_pickle(pkl_path)
    demos = _trajectories_from_payload(obj)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    for i, (sx, sy, label) in enumerate(demos):
        ax.plot(sx, sy, color=cmap(i % 10), linewidth=2.0, label=label)
        start_kw: dict = dict(s=70, c="green", edgecolors="darkgreen", linewidths=1.2, zorder=5)
        end_kw: dict = dict(s=70, c="red", edgecolors="darkred", linewidths=1.2, zorder=5)
        if i == 0:
            start_kw["label"] = "start"
            end_kw["label"] = "end"
        ax.scatter(sx[0], sy[0], **start_kw)
        ax.scatter(sx[-1], sy[-1], **end_kw)

    attractor = obj.get("attractor")
    if attractor is not None:
        att = np.asarray(attractor, dtype=np.float64).reshape(-1)
        ax.plot(att[0], att[1], "g*", markersize=15, label="attractor", zorder=6)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    base = title or os.path.splitext(os.path.basename(pkl_path))[0]
    ax.set_title(base)
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {savefig}", file=sys.stderr)
    if show:
        plt.show()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Plot or export the analytic straight-line demonstration pickle.",
    )
    p.add_argument(
        "--pkl",
        type=str,
        default=_DEFAULT_PKL,
        help=f"Input/output pickle path (default: {_DEFAULT_PKL})",
    )
    p.add_argument(
        "--mode",
        choices=("plot", "export"),
        default="plot",
        help="plot: matplotlib 2D; export: write pickle from ladder2_line demo",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=_default_dt(),
        help="Scalar timestep stored in delta_t on export",
    )
    p.add_argument("--title", type=str, default=None, help="Figure title for --mode plot")
    p.add_argument(
        "--savefig",
        type=str,
        default=None,
        metavar="OUT.png",
        help="Save matplotlib figure to this path",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive matplotlib window",
    )
    args = p.parse_args(argv)

    if args.mode == "export":
        export_analytical_pickle(args.pkl, dt=args.dt)
        return 0

    pkl_path = os.path.expanduser(args.pkl)
    if not os.path.isfile(pkl_path):
        raise SystemExit(
            f"Pickle not found: {pkl_path}\n"
            "Create it with: python myscripts/utils/plot_demo.py --mode export"
        )
    run_plot(pkl_path, title=args.title, savefig=args.savefig, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
