"""
Example code use:
```
python /home/khw/Projects/pumafabrics/pumafabrics/puma_adapted/tools/plot_3D.py \
/home/khw/Projects/pumafabrics/pumafabrics/puma_adapted/datasets/kuka/pick_tomato_31may/ee_state_0.pk
```

Plot every `*.pk` in a directory on one figure:
```
python .../plot_3D.py --folder \
/home/khw/Projects/pumafabrics/pumafabrics/puma_adapted/datasets/kuka/pick_tomato_31may
```
NOTE: Only one Dash App page can be launched at one time.
"""
import argparse
import glob
import os
import pickle
import sys

try:
    import plotly.graph_objects as go
except Exception as e:
    raise SystemExit(
        "Missing dependency: plotly.\n"
        "Install it in your active environment, e.g. `pip install plotly`, then rerun.\n"
        f"Original import error: {e}"
    )

# Dash is only used to expose relayout (camera) values interactively.
# Plotting the figure itself only requires Plotly.
try:
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
    _DASH_AVAILABLE = True
except Exception:
    _DASH_AVAILABLE = False

def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _pk_paths_in_folder(folder: str) -> list[str]:
    folder = os.path.abspath(os.path.expanduser(folder))
    paths = sorted(glob.glob(os.path.join(folder, "*.pk")))
    return paths


def _add_object_traces(fig: "go.Figure", obj, trace_label: str, debug: bool) -> None:
    """
    Append traces from one pickle object to an existing figure.
    See _make_figure_from_pickle for supported object shapes.
    """
    if isinstance(obj, dict) and "3D_plot" in obj:
        if debug:
            print(
                f"[plot_3D] {trace_label}: plotly-traces dict (key: '3D_plot')",
                file=sys.stderr,
            )
        sub = go.Figure(data=obj["3D_plot"])
        for i, tr in enumerate(sub.data):
            nm = trace_label if len(sub.data) == 1 else f"{trace_label}_{i}"
            tr.name = nm
            fig.add_trace(tr)
        return

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported pickle type: {type(obj)} (expected dict-like).")

    xyz = None
    for key in ("x_pos", "pos_fk"):
        if key in obj:
            xyz = obj[key]
            if debug:
                print(
                    f"[plot_3D] {trace_label}: trajectory dict (key: '{key}')",
                    file=sys.stderr,
                )
            break

    if xyz is None:
        available = ", ".join(sorted(obj.keys()))
        raise KeyError(
            "Could not infer what to plot.\n"
            "Expected either key `3D_plot` (plotly traces) or a trajectory key like `x_pos`/`pos_fk`.\n"
            f"Top-level keys: {available}"
        )

    xs, ys, zs = [], [], []
    for p in xyz:
        if p is None:
            continue
        if len(p) < 3:
            raise ValueError(f"Trajectory point has <3 elements: {p}")
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines+markers",
            line=dict(width=6),
            marker=dict(size=3),
            name=trace_label,
        )
    )


def _make_figure_from_pickles(paths: list[str], debug: bool) -> "go.Figure":
    """Load multiple .pk files and combine all trajectories/traces into one figure."""
    fig = go.Figure()
    for path in paths:
        obj = _load_pickle(path)
        if debug and isinstance(obj, dict):
            print(f"[plot_3D] loaded: {path}", file=sys.stderr)
            print(
                f"[plot_3D] top-level keys ({len(obj)}): {sorted(obj.keys())}",
                file=sys.stderr,
            )
        label = os.path.splitext(os.path.basename(path))[0]
        _add_object_traces(fig, obj, label, debug)
    fig.update_layout(scene=dict(aspectmode="data"))
    return fig


def _make_figure_from_pickle(obj, debug: bool) -> "go.Figure":
    """
    Supports:
    - training/eval plot pickles with a pre-built Plotly trace list at key `3D_plot`
    - demonstration/state pickles with keys like `x_pos` / `pos_fk` (list/array of xyz)
    """
    if isinstance(obj, dict) and "3D_plot" in obj:
        if debug:
            print("[plot_3D] detected format: plotly-traces dict (key: '3D_plot')", file=sys.stderr)
        return go.Figure(data=obj["3D_plot"])

    fig = go.Figure()
    _add_object_traces(fig, obj, "trajectory", debug)
    fig.update_layout(scene=dict(aspectmode="data"))
    return fig


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Visualize 3D Plotly pickles or trajectory pickles.")
    parser.add_argument(
        "pickle_path",
        nargs="?",
        default=None,
        help="Path to a .pickle/.pkl/.pk file, or with --folder a directory containing .pk files. "
        "If omitted, uses the legacy iteration-based default.",
    )
    parser.add_argument(
        "--folder",
        action="store_true",
        help="Treat pickle_path as a directory and combine all *.pk files (sorted) into one figure.",
    )
    parser.add_argument(
        "--iteration",
        default=None,
        help="Legacy mode: iteration number used to build the default training-pickle path.",
    )
    parser.add_argument("--fix_axes_limits", action="store_true", help="Set axes limits to pre-defined values (defined within script.)")
    parser.add_argument("--debug", action="store_true", help="Print inferred format and keys to stderr.")
    parser.add_argument("--no-dash", action="store_true", help="Never start Dash (just open the figure).")
    args = parser.parse_args(argv)

    if args.pickle_path is None:
        if args.iteration is None:
            # Backwards-compatible behavior with previous script: first CLI arg was `iteration`
            # (but provide a clearer error if it's missing).
            raise SystemExit(
                "No input provided.\n"
                "Usage examples:\n"
                "  python3 pumafabrics/puma_adapted/tools/plot_3D.py path/to/file.pk\n"
                "  python3 pumafabrics/puma_adapted/tools/plot_3D.py --folder path/to/dir_with_pk_files\n"
                "  python3 pumafabrics/puma_adapted/tools/plot_3D.py --iteration 0\n"
            )
        args.pickle_path = (
            "results/final/LASA_S2/1st_order_S2/22/images/primitive_0_iter_%s.pickle" % args.iteration
        )

    pickle_path = os.path.expanduser(args.pickle_path)
    if not os.path.exists(pickle_path):
        raise SystemExit(f"Path not found: {pickle_path}")

    if args.folder:
        if not os.path.isdir(pickle_path):
            raise SystemExit(f"--folder expects a directory: {pickle_path}")
        pk_paths = _pk_paths_in_folder(pickle_path)
        if not pk_paths:
            raise SystemExit(f"No .pk files found in: {pickle_path}")
        fig = _make_figure_from_pickles(pk_paths, debug=args.debug)
    else:
        if not os.path.isfile(pickle_path):
            raise SystemExit(f"Not a file (use --folder for directories): {pickle_path}")
        obj = _load_pickle(pickle_path)
        if args.debug and isinstance(obj, dict):
            print(f"[plot_3D] loaded: {pickle_path}", file=sys.stderr)
            print(f"[plot_3D] top-level keys ({len(obj)}): {sorted(obj.keys())}", file=sys.stderr)
        fig = _make_figure_from_pickle(obj, debug=args.debug)

    camera = dict(
        eye=dict(x=-0.8550474526258948, y=-0.8632259816571023, z=0.8694060571650828),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1),
    )
    if args.fix_axes_limits:
        print('Using fixed axes limits')
        fig.update_layout(
            scene=dict(
                camera=camera,
                xaxis=dict(range=[0.0, 1.0]),
                yaxis=dict(range=[-0.5, 0.5]),
                zaxis=dict(range=[0.0, 0.8]),
            )
        )
    else:
        print("Using automatic axes limits")
        fig.update_layout(
            scene=dict(
                camera=camera,
            )
        )

    fig.show()

    if args.no_dash:
        return 0

    if not _DASH_AVAILABLE:
        raise SystemExit(
            "Dash is not available (or incompatible with Werkzeug). "
            "Displayed the Plotly figure; skipping Dash relayout viewer."
        )

    app = dash.Dash()
    app.layout = html.Div(
        [
            html.Div(id="output"),  # use to print current relayout values
            dcc.Graph(id="fig", figure=fig),
        ]
    )

    @app.callback(Output("output", "children"), Input("fig", "relayoutData"))
    def show_data(data):
        # show camera settings like eye upon change
        return [str(data)]

    app.run_server(debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
