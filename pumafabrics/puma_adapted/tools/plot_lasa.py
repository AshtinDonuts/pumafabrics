"""
Load LASA HRI-style `.mat` demonstrations (variable `demos`) and either plot them in 2D
or export a pickle that `plot_3D.py` accepts (`x_pos` as Nx3 points; z=0 for planar data).

Examples:
  # Matplotlib 2D plot (all demos in the file)
  python plot_lasa.py --mat ../datasets/LASA/Angle.mat

  # By primitive name (resolves under puma_adapted/datasets/LASA/)
  python plot_lasa.py --primitive Line

  # Export demo 0 for use with plot_3D.py
  python plot_lasa.py --primitive Angle --export /tmp/angle_demo0.pk --demo-index 0

  # Then:
  python plot_3D.py /tmp/angle_demo0.pk
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

try:
    import scipy.io as sio
except ImportError as e:
    raise SystemExit(
        "scipy is required (e.g. `pip install scipy`).\n" f"Original error: {e}"
    ) from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    plt = None  # optional until --mode plot


def _default_lasa_dir() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "LASA")
    )


def _resolve_mat_path(mat: str | None, primitive: str | None) -> str:
    if mat:
        path = os.path.expanduser(mat)
        if not os.path.isfile(path):
            # Allow primitive-style name without extension
            if not path.endswith(".mat"):
                cand = path + ".mat"
                if os.path.isfile(cand):
                    return cand
            raise FileNotFoundError(f"Not a file: {path}")
        return path
    if primitive:
        base = _default_lasa_dir()
        name = primitive if primitive.endswith(".mat") else primitive + ".mat"
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected LASA file at: {path}")
        return path
    raise ValueError("Provide --mat PATH or --primitive NAME")


def load_lasa_demonstrations(mat_path: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float]]:
    """Returns list of (x, y) arrays per demo and list of scalar dt per demo."""
    mat_file = sio.loadmat(mat_path)
    if "demos" not in mat_file:
        raise KeyError(f"No 'demos' variable in {mat_path}; keys: {sorted(mat_file.keys())}")
    data = mat_file["demos"]
    demos: list[tuple[np.ndarray, np.ndarray]] = []
    dts: list[float] = []
    for j in range(data.shape[1]):
        s_x = np.asarray(data[0, j]["pos"][0, 0][0]).astype(np.float64).reshape(-1)
        s_y = np.asarray(data[0, j]["pos"][0, 0][1]).astype(np.float64).reshape(-1)
        demos.append((s_x, s_y))
        dts.append(float(data[0, j]["dt"][0, 0][0, 0]))
    return demos, dts


def demo_to_plot3d_dict(
    sx: np.ndarray,
    sy: np.ndarray,
    dt: float,
    source: str,
) -> dict:
    """Build a dict compatible with plot_3D.py (trajectory key `x_pos`)."""
    sx = np.asarray(sx, dtype=np.float64).reshape(-1)
    sy = np.asarray(sy, dtype=np.float64).reshape(-1)
    n = int(min(sx.size, sy.size))
    x_pos = [[float(sx[i]), float(sy[i]), 0.0] for i in range(n)]
    delta_t = np.full(n, float(dt), dtype=np.float64)
    return {
        "x_pos": x_pos,
        "delta_t": delta_t,
        "source": source,
        "format": "lasa_export_for_plot_3D",
    }


def run_plot(mat_path: str, title: str | None, savefig: str | None) -> None:
    if plt is None:
        raise SystemExit("matplotlib is required for --mode plot (`pip install matplotlib`).")
    demos, _dts = load_lasa_demonstrations(mat_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    for i, (sx, sy) in enumerate(demos):
        ax.plot(sx, sy, color=cmap(i % 10), linewidth=2.0, label=f"demo {i}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    base = title or os.path.splitext(os.path.basename(mat_path))[0]
    ax.set_title(base)
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {savefig}", file=sys.stderr)
    plt.show()


def run_export(mat_path: str, demo_index: int, out_path: str) -> None:
    demos, dts = load_lasa_demonstrations(mat_path)
    if demo_index < 0 or demo_index >= len(demos):
        raise IndexError(f"demo_index {demo_index} out of range [0, {len(demos) - 1}]")
    sx, sy = demos[demo_index]
    payload = demo_to_plot3d_dict(sx, sy, dts[demo_index], source=os.path.abspath(mat_path))
    out_path = os.path.expanduser(out_path)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Wrote {out_path} ({len(payload['x_pos'])} points, plot_3D.py key: x_pos)", file=sys.stderr)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Plot or export LASA .mat demonstrations.")
    p.add_argument("--mat", type=str, default=None, help="Path to a LASA .mat file")
    p.add_argument(
        "--primitive",
        type=str,
        default=None,
        help="Primitive name without path (loads from puma_adapted/datasets/LASA/<name>.mat)",
    )
    p.add_argument(
        "--mode",
        choices=("plot", "export"),
        default="plot",
        help="plot: matplotlib 2D; export: pickle for plot_3D.py",
    )
    p.add_argument("--demo-index", type=int, default=0, help="Demo index for --mode export")
    p.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="OUT.pk",
        help="Output pickle path (required for --mode export)",
    )
    p.add_argument("--title", type=str, default=None, help="Figure title for --mode plot")
    p.add_argument(
        "--savefig",
        type=str,
        default=None,
        metavar="OUT.png",
        help="Save matplotlib figure to this path (still shows window if display available)",
    )
    args = p.parse_args(argv)

    mat_path = _resolve_mat_path(args.mat, args.primitive)

    if args.mode == "export":
        if not args.export:
            p.error("--mode export requires --export OUT.pk")
        run_export(mat_path, args.demo_index, args.export)
        return 0

    run_plot(mat_path, args.title, args.savefig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
