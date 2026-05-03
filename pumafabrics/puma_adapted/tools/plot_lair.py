"""
Load LAIR `.npy` demonstrations (same layout as `load_numpy_file` in data_preprocessing):
arrays shaped `(1, T, 2)` or `(T, 2)` with planar trajectory points.

Layout on disk: `puma_adapted/datasets/LAIR/<primitive>/*.npy`

Either plot trajectories in 2D (matplotlib) or export a pickle for `plot_3D.py`
(`x_pos` as [x, y, 0] per step; `delta_t` defaults to 0.03 like the training loader).

Examples:
  python plot_lair.py --primitive e

  python plot_lair.py --npy ../datasets/LAIR/e/e_0.npy

  python plot_lair.py --primitive double_loop --demo-index 0 --mode export --export /tmp/lair_dl_0.pk
  python plot_3D.py /tmp/lair_dl_0.pk
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _default_lair_dir() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "LAIR")
    )


def _npy_sort_key(filename: str) -> tuple[str, int]:
    """Sort `e_0.npy`, `G_angle_12.npy` numerically by trailing index."""
    base = filename[:-4] if filename.endswith(".npy") else filename
    m = re.search(r"_(\d+)$", base)
    if m:
        return (base[: m.start()], int(m.group(1)))
    return (base, 0)


def _sorted_npy_paths(primitive_dir: str) -> list[str]:
    files = [f for f in os.listdir(primitive_dir) if f.endswith(".npy")]
    files.sort(key=_npy_sort_key)
    return [os.path.join(primitive_dir, f) for f in files]


def load_lair_npy(npy_path: str) -> np.ndarray:
    """
    Returns (T, D) with D in {2, 3}, matching `load_numpy_file` squeeze rules.
    """
    data = np.load(os.path.expanduser(npy_path))
    if data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze in {npy_path}, got shape {data.shape}")
    return np.asarray(data, dtype=np.float64)


def trajectory_to_xy(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pts.shape[1] < 2:
        raise ValueError(f"Need at least 2 columns, got shape {pts.shape}")
    return pts[:, 0], pts[:, 1]


def demo_to_plot3d_dict(pts_txd: np.ndarray, dt: float, source: str) -> dict:
    """Build dict for `plot_3D.py` (`x_pos` as 3-vectors)."""
    t = pts_txd.shape[0]
    if pts_txd.shape[1] == 2:
        x_pos = [[float(pts_txd[i, 0]), float(pts_txd[i, 1]), 0.0] for i in range(t)]
    elif pts_txd.shape[1] == 3:
        x_pos = [
            [float(pts_txd[i, 0]), float(pts_txd[i, 1]), float(pts_txd[i, 2])] for i in range(t)
        ]
    else:
        raise ValueError(f"Unsupported D={pts_txd.shape[1]} for plot_3D export")
    delta_t = np.full(t, float(dt), dtype=np.float64)
    return {
        "x_pos": x_pos,
        "delta_t": delta_t,
        "source": source,
        "format": "lair_export_for_plot_3D",
    }


def _resolve_inputs(
    npy: str | None, primitive: str | None
) -> tuple[list[str], str]:
    """
    Returns (list of npy paths to use, label for title).
    """
    if npy:
        path = os.path.expanduser(npy)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Not a file: {path}")
        return [path], os.path.splitext(os.path.basename(path))[0]
    if primitive:
        sub = os.path.join(_default_lair_dir(), primitive)
        if not os.path.isdir(sub):
            raise FileNotFoundError(f"Not a directory: {sub}")
        paths = _sorted_npy_paths(sub)
        if not paths:
            raise FileNotFoundError(f"No .npy files in {sub}")
        return paths, primitive
    raise ValueError("Provide --npy PATH or --primitive NAME (e.g. e, double_loop)")


def run_plot(
    paths: list[str],
    title: str | None,
    savefig: str | None,
    demo_index: int | None,
) -> None:
    if plt is None:
        raise SystemExit("matplotlib is required for --mode plot (`pip install matplotlib`).")
    if demo_index is not None:
        if demo_index < 0 or demo_index >= len(paths):
            raise IndexError(f"demo_index {demo_index} out of range [0, {len(paths) - 1}]")
        paths = [paths[demo_index]]

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    for i, pth in enumerate(paths):
        pts = load_lair_npy(pth)
        sx, sy = trajectory_to_xy(pts)
        label = os.path.basename(pth) if len(paths) > 1 else f"demo ({os.path.basename(pth)})"
        ax.plot(sx, sy, color=cmap(i % 10), linewidth=2.0, label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if len(paths) > 1 or demo_index is not None:
        ax.legend(loc="best", fontsize=8)
    base = title or (os.path.basename(os.path.dirname(paths[0])) if len(paths) == 1 else "LAIR")
    ax.set_title(base)
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {savefig}", file=sys.stderr)
    plt.show()


def run_export(paths: list[str], demo_index: int, out_path: str, dt: float) -> None:
    if demo_index < 0 or demo_index >= len(paths):
        raise IndexError(f"demo_index {demo_index} out of range [0, {len(paths) - 1}]")
    pth = paths[demo_index]
    pts = load_lair_npy(pth)
    payload = demo_to_plot3d_dict(pts, dt=dt, source=os.path.abspath(pth))
    out_path = os.path.expanduser(out_path)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(
        f"Wrote {out_path} ({len(payload['x_pos'])} points, plot_3D.py key: x_pos)",
        file=sys.stderr,
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Plot or export LAIR .npy demonstrations.")
    p.add_argument("--npy", type=str, default=None, help="Path to a single .npy trajectory file")
    p.add_argument(
        "--primitive",
        type=str,
        default=None,
        help="Primitive folder under datasets/LAIR/ (e.g. e, double_loop, G_angle)",
    )
    p.add_argument(
        "--mode",
        choices=("plot", "export"),
        default="plot",
        help="plot: matplotlib 2D; export: pickle for plot_3D.py",
    )
    p.add_argument(
        "--demo-index",
        type=int,
        default=None,
        help="Index into sorted .npy list for export, or plot only this demo when using --primitive",
    )
    p.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="OUT.pk",
        help="Output pickle path (required for --mode export)",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.03,
        help="Scalar timestep for delta_t in export (matches training loader dummy dt)",
    )
    p.add_argument("--title", type=str, default=None, help="Figure title for --mode plot")
    p.add_argument(
        "--savefig",
        type=str,
        default=None,
        metavar="OUT.png",
        help="Save matplotlib figure to this path",
    )
    args = p.parse_args(argv)

    paths, default_title = _resolve_inputs(args.npy, args.primitive)

    if args.mode == "export":
        if not args.export:
            p.error("--mode export requires --export OUT.pk")
        if args.npy:
            run_export(paths, 0, args.export, dt=args.dt)
        else:
            idx = args.demo_index if args.demo_index is not None else 0
            run_export(paths, idx, args.export, dt=args.dt)
        return 0

    title = args.title or default_title
    run_plot(paths, title, args.savefig, demo_index=args.demo_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
