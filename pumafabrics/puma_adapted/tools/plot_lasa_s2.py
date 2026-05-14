"""
Load LASA-on-S² JSON demonstrations (`xyz` key) and either plot them on the unit sphere
or export a pickle compatible with `plot_3D.py` (`x_pos` as unit-direction 3-vectors).

Data layout matches `load_LASA_S2` in `data_preprocessing/data_loader.py`.

Examples:
  python plot_lasa_s2.py --primitive S2_Line

  python plot_lasa_s2.py --txt ../datasets/LASA_S2/S2_Angle.txt --mode plot

  python plot_lasa_s2.py --primitive S2_Angle --mode export --export /tmp/s2_angle_0.pk --demo-index 0
  python plot_3D.py /tmp/s2_angle_0.pk
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
except ImportError:
    plt = None


def _default_lasa_s2_dir() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "LASA_S2")
    )


def _resolve_txt_path(txt: str | None, primitive: str | None) -> str:
    if txt:
        path = os.path.expanduser(txt)
        if not os.path.isfile(path):
            if not path.endswith(".txt"):
                cand = path + ".txt"
                if os.path.isfile(cand):
                    return cand
            raise FileNotFoundError(f"Not a file: {path}")
        return path
    if primitive:
        base = _default_lasa_s2_dir()
        name = primitive if primitive.endswith(".txt") else primitive + ".txt"
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected LASA_S2 file at: {path}")
        return path
    raise ValueError("Provide --txt PATH or --primitive NAME (e.g. S2_Line)")


def load_lasa_s2_demonstrations(txt_path: str) -> list[np.ndarray]:
    """
    Returns one (T, 3) float array per demonstration; rows are unit-sphere points in R^3.
    """
    with open(txt_path, encoding="utf-8") as f:
        raw = f.read()
    data = json.loads(raw)
    if "xyz" not in data:
        raise KeyError(f"No 'xyz' in {txt_path}; keys: {sorted(data.keys())}")
    demos: list[np.ndarray] = []
    for j, demo in enumerate(data["xyz"]):
        arr = np.asarray(demo, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Demo {j}: expected shape (T, 3), got {arr.shape}")
        demos.append(arr)
    return demos


def demo_to_plot3d_dict(points_tx3: np.ndarray, dt: float, source: str) -> dict:
    """Dict compatible with plot_3D.py trajectory mode (`x_pos` list of [x,y,z])."""
    n = points_tx3.shape[0]
    x_pos = [[float(points_tx3[i, 0]), float(points_tx3[i, 1]), float(points_tx3[i, 2])] for i in range(n)]
    delta_t = np.full(n, float(dt), dtype=np.float64)
    return {
        "x_pos": x_pos,
        "delta_t": delta_t,
        "source": source,
        "format": "lasa_s2_export_for_plot_3D",
    }


def _add_unit_sphere(ax, n_u: int = 24, n_v: int = 16, color: str = "0.75", alpha: float = 0.15) -> None:
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0, antialiased=True, shade=True)


def run_plot(txt_path: str, title: str | None, savefig: str | None, wireframe: bool) -> None:
    if plt is None:
        raise SystemExit("matplotlib is required for --mode plot (`pip install matplotlib`).")
    demos = load_lasa_s2_demonstrations(txt_path)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    if wireframe:
        u = np.linspace(0, 2 * np.pi, 32)
        v = np.linspace(0, np.pi, 16)
        uu, vv = np.meshgrid(u, v)
        xs = np.cos(uu) * np.sin(vv)
        ys = np.sin(uu) * np.sin(vv)
        zs = np.cos(vv)
        ax.plot_wireframe(xs, ys, zs, color="gray", linewidth=0.3, alpha=0.35)
    else:
        _add_unit_sphere(ax)

    cmap = plt.get_cmap("tab10")
    for i, pts in enumerate(demos):
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=cmap(i % 10), linewidth=2.0, label=f"demo {i}")
        start_kw: dict = dict(s=70, c="green", edgecolors="darkgreen", linewidths=1.2, zorder=5)
        end_kw: dict = dict(s=70, c="red", edgecolors="darkred", linewidths=1.2, zorder=5)
        if i == 0:
            start_kw["label"] = "start"
            end_kw["label"] = "end"
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], **start_kw)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], **end_kw)

    lim = 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 1.0))
    base = title or os.path.splitext(os.path.basename(txt_path))[0]
    ax.set_title(base)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {savefig}", file=sys.stderr)
    plt.show()


def run_export(txt_path: str, demo_index: int, out_path: str, dt: float) -> None:
    demos = load_lasa_s2_demonstrations(txt_path)
    if demo_index < 0 or demo_index >= len(demos):
        raise IndexError(f"demo_index {demo_index} out of range [0, {len(demos) - 1}]")
    payload = demo_to_plot3d_dict(demos[demo_index], dt=dt, source=os.path.abspath(txt_path))
    out_path = os.path.expanduser(out_path)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(
        f"Wrote {out_path} ({len(payload['x_pos'])} points, plot_3D.py key: x_pos)",
        file=sys.stderr,
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Plot or export LASA_S2 spherical demonstrations.")
    p.add_argument("--txt", type=str, default=None, help="Path to a LASA_S2 .txt (JSON) file")
    p.add_argument(
        "--primitive",
        type=str,
        default=None,
        help="Primitive name, e.g. S2_Line (loads puma_adapted/datasets/LASA_S2/<name>.txt)",
    )
    p.add_argument(
        "--mode",
        choices=("plot", "export"),
        default="plot",
        help="plot: matplotlib 3D on S^2; export: pickle for plot_3D.py",
    )
    p.add_argument("--demo-index", type=int, default=0, help="Demo index for --mode export")
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
        default=1.0,
        help="Scalar timestep written to delta_t (matches training loader default)",
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
        "--wireframe",
        action="store_true",
        help="Use a light wireframe sphere instead of a shaded surface",
    )
    args = p.parse_args(argv)

    txt_path = _resolve_txt_path(args.txt, args.primitive)

    if args.mode == "export":
        if not args.export:
            p.error("--mode export requires --export OUT.pk")
        run_export(txt_path, args.demo_index, args.export, dt=args.dt)
        return 0

    run_plot(txt_path, args.title, args.savefig, wireframe=args.wireframe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
