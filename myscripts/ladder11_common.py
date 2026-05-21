"""
Shared utilities for ladder step 11: demo-curve-to-demo-curve policy transport.

Extends the policy_transportation framework (ladder 8) by fitting the transport map
directly from a source system's demonstration curve to a *real* target curve drawn from
the LASA or LAIR datasets.  The procedure mirrors PolicyTransportation.fit():

    1. Fit affine gamma(x) = M x + b  (Procrustes from resampled source/target keypoints).
    2. Compute residuals: delta = target - gamma(source).
    3. Fit nonlinear psi(z) as Gaussian-RBF interpolant on (gamma(source), delta).
    4. Evaluate phi(x) = gamma(x) + psi(gamma(x)),   v_hat = J_phi(x) @ v.

The RBF interpolant (RBFResidualRegressor2D) replaces the simple parametric
ParametricDeltaRegressor2D from ladder 8 so that the map can capture the full shape
deformation between arbitrary demo curves.

Reference:
    policy_transportation.transportation.PolicyTransportation
    https://github.com/franzesegiovanni/policy_transportation/blob/devel/
        policy_transportation/transportation/transportation.py
"""

from __future__ import annotations

import os
import sys
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pumafabrics.puma_adapted.tools.animation import TrajectoryPlotter

_MYSCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _MYSCRIPTS not in sys.path:
    sys.path.insert(0, _MYSCRIPTS)

from ladder8_common import (
    AffineTransform2D,
    PolicyTransportation2D,
    _as_batch,
    _resample_polyline,
    plot_title as _ladder8_plot_title,
)
from ladder5_common import (
    SOURCE_CHOICES,
    SOURCE_LABELS,
    SourceTag,
    _demonstration_xy,
    _load_learned_source,
)

OrderTag = Literal["1st", "2nd"]

_LASA_DIR = os.path.normpath(
    os.path.join(_REPO_ROOT, "pumafabrics", "puma_adapted", "datasets", "LASA")
)
_LAIR_DIR = os.path.normpath(
    os.path.join(_REPO_ROOT, "pumafabrics", "puma_adapted", "datasets", "LAIR")
)

TARGET_DATASET_CHOICES = ("lasa", "lair")

_LASA_DEFAULT = "heee"
_LAIR_DEFAULT = "capricorn"


# ---------------------------------------------------------------------------
# Gaussian-RBF residual regressor
# ---------------------------------------------------------------------------

class RBFResidualRegressor2D:
    """
    Nonlinear residual psi(z) fitted via Gaussian RBF interpolation.

    Interface mirrors ParametricDeltaRegressor2D so it can be dropped into
    PolicyTransportation2D.nonlinear_transform.

    Given keypoint pairs (z_i, delta_i) the map is:

        psi(z) = sum_i W_i * exp(-||z - z_i||^2 / (2 h^2))

    where h is the bandwidth (auto-set to the median pairwise distance if None),
    and W is the (n_centers, 2) weight matrix solving K W = delta with Tikhonov
    regularisation (strength `reg`).
    """

    def __init__(self, bandwidth: float | None = None, reg: float = 1e-6) -> None:
        self.bandwidth = bandwidth
        self.reg = float(reg)
        self._centers: np.ndarray | None = None
        self._weights: np.ndarray | None = None  # (n_centers, 2)
        self._h: float = 1.0

    # ------------------------------------------------------------------
    # Internal kernel helpers
    # ------------------------------------------------------------------

    def _rbf(self, dists_sq: np.ndarray) -> np.ndarray:
        return np.exp(-dists_sq / (2.0 * self._h ** 2))

    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """K[i, j] = rbf(||X_i - Y_j||^2), shape (|X|, |Y|)."""
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (|X|, |Y|, 2)
        return self._rbf((diff ** 2).sum(axis=-1))

    # ------------------------------------------------------------------
    # Fit / predict / derivative
    # ------------------------------------------------------------------

    def fit(self, inputs: np.ndarray, deltas: np.ndarray) -> None:
        inputs = _as_batch(inputs)
        deltas = _as_batch(deltas)
        self._centers = inputs.copy()

        if self.bandwidth is None:
            diffs = inputs[:, np.newaxis, :] - inputs[np.newaxis, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=-1))
            med = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
            self._h = max(med, 1e-8)
        else:
            self._h = max(float(self.bandwidth), 1e-8)

        K = self._gram(inputs, inputs)
        reg_mat = self.reg * np.eye(K.shape[0])
        self._weights = np.linalg.solve(K + reg_mat, deltas)  # (n_centers, 2)

    def predict(self, pos: np.ndarray, *, return_std: bool = False):
        pos = _as_batch(pos)
        K = self._gram(pos, self._centers)
        mean = K @ self._weights  # (N, 2)
        if return_std:
            return mean, np.zeros_like(mean)
        return mean

    def derivative(self, pos: np.ndarray, *, return_var: bool = False):
        """
        Analytical Jacobian of psi wrt pos, shape (N, 2, 2).

        d(psi_d)/d(pos_j) = sum_i W_id * (-1/h^2) * (pos_j - c_ij) * K[pos, c_i]
        """
        pos = _as_batch(pos)
        diff = pos[:, np.newaxis, :] - self._centers[np.newaxis, :, :]  # (N, n_c, 2)
        K = self._rbf((diff ** 2).sum(axis=-1))          # (N, n_c)
        scale = -1.0 / (self._h ** 2)
        grad_K = scale * diff * K[:, :, np.newaxis]      # (N, n_c, 2)
        # J[i, d, j] = einsum('kd, ikj -> idj', W, grad_K)
        J = np.einsum("kd,ikj->idj", self._weights, grad_K)  # (N, 2, 2)
        if return_var:
            return J, np.zeros_like(J)
        return J


# ---------------------------------------------------------------------------
# Improved transport class with affine-initialized Newton inverse
# ---------------------------------------------------------------------------

class PolicyTransportation2D_RBF(PolicyTransportation2D):
    """
    PolicyTransportation2D extended with a numerically robust inverse.

    The base class inverse_position uses undamped fixed-point iteration starting
    from x=y.  That diverges when the map has large scale (e.g. source in [-1,1]
    and target in LASA millimetre space).  Here we instead initialize from the
    affine pseudo-inverse and refine with damped Newton steps.
    """

    def inverse_position(
        self,
        y: np.ndarray,
        *,
        max_iter: int = 50,
        step_size: float = 0.5,
        tol: float = 1e-7,
    ) -> np.ndarray:
        y = _as_batch(y)
        # Analytic affine inverse as warm start
        try:
            M_inv = np.linalg.inv(self.affine_transform.matrix)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(self.affine_transform.matrix)
        x = ((y - self.affine_transform.translation) @ M_inv.T)

        for _ in range(max_iter):
            residual = self.transport(x, return_std=False) - y
            if float(np.max(np.abs(residual))) < tol:
                break
            J = self.compute_jacobian(x, return_var=False)
            dx = np.linalg.solve(J, residual[..., np.newaxis])[..., 0]
            x = x - step_size * dx
        return x


def build_policy_transport_ds(
    order,
    source,
    transport: PolicyTransportation2D_RBF,
    *,
    attractor,
    gain: float = 1.0,
    kp: float = 1.0,
    kd=None,
    dt: float = 0.01,
    delta_t: float = 0.1,
):
    """Wrapper that passes ladder-11 transport (with robust inverse) to the DS builders."""
    from ladder8_common import (
        PolicyTransportedFirstOrderDS,
        PolicyTransportedSecondOrderDS,
    )
    from ladder5_common import _load_learned_source

    learner, data = None, None
    if source != "analytic":
        learner, _, data, _ = _load_learned_source(source, order)

    if order == "1st":
        return PolicyTransportedFirstOrderDS(
            source,
            transport,
            attractor=attractor,
            gain=gain,
            learner=learner,
            data=data,
            dt=dt,
            delta_t=delta_t,
        ), data
    return PolicyTransportedSecondOrderDS(
        source,
        transport,
        attractor=attractor,
        kp=kp,
        kd=kd,
        learner=learner,
        data=data,
        dt=dt,
        delta_t=delta_t,
    ), data


# ---------------------------------------------------------------------------
# Dataset loaders (LASA / LAIR)
# ---------------------------------------------------------------------------

def lasa_names() -> list[str]:
    """Return available LASA dataset names (without .mat extension)."""
    if not os.path.isdir(_LASA_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(_LASA_DIR)
        if f.endswith(".mat")
    )


def lair_names() -> list[str]:
    """Return available LAIR dataset names (subdirectory names)."""
    if not os.path.isdir(_LAIR_DIR):
        return []
    return sorted(
        d for d in os.listdir(_LAIR_DIR) if os.path.isdir(os.path.join(_LAIR_DIR, d))
    )


def load_lasa_curve_xy(name: str, demo_index: int = 0) -> np.ndarray:
    """
    Load one LASA demonstration as (T, 2) positions.

    Parameters
    ----------
    name:
        Dataset name without extension, e.g. ``"heee"``, ``"CShape"``.
    demo_index:
        Index of the demonstration to use (0-based).
    """
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            "scipy is required to read LASA .mat files (pip install scipy)."
        ) from exc

    mat_path = os.path.join(_LASA_DIR, f"{name}.mat")
    if not os.path.isfile(mat_path):
        avail = lasa_names()
        raise FileNotFoundError(
            f"LASA dataset '{name}' not found at {mat_path}.\n"
            f"Available: {avail}"
        )
    mat = sio.loadmat(mat_path)
    if "demos" not in mat:
        raise KeyError(f"No 'demos' key in {mat_path}")
    demos = mat["demos"]
    n_demos = int(demos.shape[1])
    if demo_index < 0 or demo_index >= n_demos:
        raise IndexError(
            f"demo_index {demo_index} out of range for {n_demos} demos in '{name}'."
        )
    ep = demos[0, demo_index]
    pos = np.asarray(ep["pos"][0, 0], dtype=float)  # (2, T)
    return pos.T  # (T, 2)


def load_lair_curve_xy(name: str, episode_index: int = 0) -> np.ndarray:
    """
    Load one LAIR demonstration as (T, 2) positions.

    Parameters
    ----------
    name:
        Subdirectory name under LAIR/, e.g. ``"capricorn"``, ``"e"``.
    episode_index:
        Episode/file index (0-based).  The file is expected at
        ``LAIR/<name>/<name>_<episode_index>.npy``.
    """
    npy_path = os.path.join(_LAIR_DIR, name, f"{name}_{episode_index}.npy")
    if not os.path.isfile(npy_path):
        avail = lair_names()
        raise FileNotFoundError(
            f"LAIR episode '{name}/{episode_index}' not found at {npy_path}.\n"
            f"Available subdirectories: {avail}"
        )
    arr = np.load(npy_path).astype(float)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # (1, T, 2) -> (T, 2)
    return arr  # (T, 2)


def load_target_curve_xy(
    dataset: str,
    name: str,
    demo_index: int = 0,
) -> np.ndarray:
    """
    Unified loader: ``dataset`` is ``'lasa'`` or ``'lair'``.

    Returns
    -------
    np.ndarray of shape (T, 2).
    """
    ds = dataset.lower()
    if ds == "lasa":
        return load_lasa_curve_xy(name, demo_index)
    if ds == "lair":
        return load_lair_curve_xy(name, demo_index)
    raise ValueError(f"Unknown dataset '{dataset}'. Choose 'lasa' or 'lair'.")


# ---------------------------------------------------------------------------
# Transport map construction from demo curves
# ---------------------------------------------------------------------------

def source_keypoints_for_fit(
    source: SourceTag,
    data: dict | None,
    *,
    n_pts: int = 40,
) -> np.ndarray:
    """
    Return (n_pts, 2) keypoints sampled from the source system's demonstration.

    For the analytic source there is no demo; a synthetic straight-line is used.
    """
    from ladder8_common import source_keypoints_for_fit as _base
    return _base(source, data, n_pts=n_pts)


def build_transport_from_demos(
    source: SourceTag,
    source_data: dict | None,
    target_xy: np.ndarray,
    *,
    n_keypoints: int = 40,
    do_scale: bool = True,
    do_rotation: bool = True,
    rbf_bandwidth: float | None = None,
    rbf_reg: float = 1e-6,
) -> PolicyTransportation2D:
    """
    Fit a PolicyTransportation2D map from source-system keypoints to a target curve.

    Parameters
    ----------
    source:
        Source system tag (``'analytic'``, ``'learned2'``, ``'learned3'``, ``'learned4'``).
    source_data:
        PUMA data dict for learned sources (from ``_load_learned_source``); ``None`` for analytic.
    target_xy:
        Raw (T, 2) target demonstration curve (will be resampled to ``n_keypoints``).
    n_keypoints:
        Number of keypoints to resample both curves to before fitting.
    do_scale:
        Whether to include scale in the affine part.
    do_rotation:
        Whether to include rotation in the affine part.
    rbf_bandwidth:
        Bandwidth h for the Gaussian RBF.  ``None`` = median-heuristic auto-setting.
    rbf_reg:
        Tikhonov regularisation for the RBF linear solve.

    Returns
    -------
    Fitted PolicyTransportation2D instance.
    """
    source_pts = source_keypoints_for_fit(source, source_data, n_pts=n_keypoints)
    target_pts = _resample_polyline(np.asarray(target_xy, dtype=float), n_keypoints)

    transport = PolicyTransportation2D_RBF(is_residual=True)
    transport.nonlinear_transform = RBFResidualRegressor2D(
        bandwidth=rbf_bandwidth, reg=rbf_reg
    )
    transport.fit(
        source_pts,
        target_pts,
        do_scale=do_scale,
        do_rotation=do_rotation,
    )
    return transport


# ---------------------------------------------------------------------------
# Initial states in the target frame (demo workspace, not transported ladder-1 grid)
# ---------------------------------------------------------------------------

def target_workspace_initial_positions(
    target_xy: np.ndarray,
    *,
    nx: int = 4,
    ny: int = 3,
    margin: float = 0.12,
) -> np.ndarray:
    """
    Grid of 2D positions inside the target demonstration bounding box.

    Same layout as ladder 3 ``workspace_initial_positions`` (4×3 → 12 points).
    Placing initials directly in target space avoids RBF extrapolation blow-up when
    mapping the small ladder-1 ``[-0.9, 0.9]`` grid through a demo-to-demo fit.
    """
    target_xy = np.asarray(target_xy, dtype=float)
    x_min = target_xy.min(axis=0)
    x_max = target_xy.max(axis=0)
    span = x_max - x_min
    lo = x_min + margin * span
    hi = x_max - margin * span
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def initial_states_for_demo_transport(
    order: OrderTag,
    target_xy: np.ndarray,
    *,
    nx: int = 4,
    ny: int = 3,
    margin: float = 0.12,
) -> np.ndarray:
    """Initial states for ladder 11: uniform grid over the target demo workspace."""
    positions = target_workspace_initial_positions(
        target_xy, nx=nx, ny=ny, margin=margin
    )
    if order == "1st":
        return positions
    return np.hstack([positions, np.zeros_like(positions)])


def _log_initial_grid_coverage(
    init_positions: np.ndarray,
    target_xy: np.ndarray,
    *,
    margin: float = 0.12,
) -> None:
    """Print whether simulation initials lie in the target demo workspace."""
    target_xy = np.asarray(target_xy, dtype=float)
    x_min = target_xy.min(axis=0)
    x_max = target_xy.max(axis=0)
    span = x_max - x_min
    lo = x_min + margin * span
    hi = x_max + margin * span
    inside = np.all((init_positions >= lo) & (init_positions <= hi), axis=1)
    n_in = int(inside.sum())
    n_tot = init_positions.shape[0]
    print(
        f"Initial grid (target frame): {n_tot} points, "
        f"x=[{init_positions[:, 0].min():.2f}, {init_positions[:, 0].max():.2f}], "
        f"y=[{init_positions[:, 1].min():.2f}, {init_positions[:, 1].max():.2f}]"
    )
    print(
        f"Target demo bbox (+{100 * margin:.0f}% margin): "
        f"x=[{lo[0]:.2f}, {hi[0]:.2f}], y=[{lo[1]:.2f}, {hi[1]:.2f}] — "
        f"{n_in}/{n_tot} grid points inside"
    )
    if n_in < n_tot:
        print(
            "Warning: some initial states lie outside the target workspace; "
            "RBF transport may be poorly conditioned there."
        )


# ---------------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------------

def plot_title_11(
    order: OrderTag,
    source: SourceTag,
    dataset: str,
    name: str,
    demo_index: int,
    transport: PolicyTransportation2D,
) -> str:
    order_label = "1st-order" if order == "1st" else "2nd-order"
    acc = transport.accuracy
    acc_str = f"{acc:.4g}" if acc is not None else "n/a"
    return (
        f"Ladder 11 — Demo-transport {order_label} DS\n"
        f"Source: {SOURCE_LABELS[source]}   →   "
        f"Target: {dataset.upper()} {name} [demo {demo_index}]   "
        f"(fit RMSE {acc_str})"
    )


def plot_trajectories_demo_transport(
    visited_pos: np.ndarray,
    attractor: np.ndarray,
    title: str,
    *,
    source_demo_xy: np.ndarray | None = None,
    target_demo_xy: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    if source_demo_xy is not None:
        ax.plot(
            source_demo_xy[:, 0],
            source_demo_xy[:, 1],
            color="steelblue",
            linewidth=3,
            linestyle="--",
            alpha=0.6,
            label="Source demo",
            zorder=4,
        )
    if target_demo_xy is not None:
        ax.plot(
            target_demo_xy[:, 0],
            target_demo_xy[:, 1],
            color="lightgray",
            linewidth=6,
            label="Target demo",
            zorder=5,
        )

    n_traj = visited_pos.shape[1]
    cmap = plt.get_cmap("gist_rainbow")
    for i in range(n_traj):
        color = cmap(i / max(n_traj - 1, 1))
        ax.plot(
            visited_pos[:, i, 0],
            visited_pos[:, i, 1],
            color=color,
            linewidth=2.0,
            zorder=10,
        )
    ax.plot(attractor[0], attractor[1], "g*", markersize=15, label="Attractor (transported)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_demo_transport_pipeline(
    order: OrderTag,
    *,
    source: SourceTag,
    target_dataset: str,
    target_name: str,
    target_demo: int = 0,
    n_keypoints: int = 40,
    do_scale: bool = True,
    do_rotation: bool = True,
    rbf_bandwidth: float | None = None,
    rbf_reg: float = 1e-6,
    grid_nx: int = 4,
    grid_ny: int = 3,
    grid_margin: float = 0.12,
    attractor: np.ndarray,
    gain: float = 1.0,
    kp: float = 1.0,
    kd: float | None = None,
    dt: float = 0.01,
    delta_t: float = 0.1,
    n_steps: int = 2000,
    live_plot: bool = True,
    pause_time: float = 1e-5,
    save_path: str | None = None,
) -> dict:
    """
    Full ladder-11 pipeline: fit transport from source demo to target demo curve,
    simulate the transported DS, and plot.

    Returns a dict with keys ``visited``, ``visited_pos``, ``goal``,
    ``source_demo``, ``target_demo``, ``transport``.
    """
    # ---- load source data (if learned) -----------------------------------
    source_data = None
    if source != "analytic":
        _, _, source_data, _ = _load_learned_source(source, order)

    # ---- load target curve -----------------------------------------------
    target_xy = load_target_curve_xy(target_dataset, target_name, target_demo)
    print(
        f"Loaded target curve: {target_dataset.upper()} '{target_name}' demo {target_demo} "
        f"({target_xy.shape[0]} pts)"
    )

    # ---- fit transport map -----------------------------------------------
    transport = build_transport_from_demos(
        source,
        source_data,
        target_xy,
        n_keypoints=n_keypoints,
        do_scale=do_scale,
        do_rotation=do_rotation,
        rbf_bandwidth=rbf_bandwidth,
        rbf_reg=rbf_reg,
    )
    rbf_h_str = "auto" if rbf_bandwidth is None else f"{rbf_bandwidth:.3g}"
    print(
        f"Transport fit RMSE: {transport.accuracy:.4e}  "
        f"(affine do_scale={do_scale}, do_rotation={do_rotation}, "
        f"RBF h={rbf_h_str})"
    )

    # ---- build transported DS and simulate --------------------------------
    ds, source_data = build_policy_transport_ds(
        order,
        source,
        transport,
        attractor=np.asarray(attractor, dtype=float),
        gain=gain,
        kp=kp,
        kd=kd,
        dt=dt,
        delta_t=delta_t,
    )

    state_init = initial_states_for_demo_transport(
        order,
        target_xy,
        nx=grid_nx,
        ny=grid_ny,
        margin=grid_margin,
    )
    _log_initial_grid_coverage(
        state_init if order == "1st" else state_init[:, :2],
        target_xy,
        margin=grid_margin,
    )

    visited = ds.simulate(state_init, n_steps)
    visited_pos = visited if order == "1st" else ds.positions(visited)

    goal_tgt = ds.attractor_target

    # ---- demo curves for overlay -----------------------------------------
    source_demo = _demonstration_xy(source, source_data)
    target_demo_xy_resampled = _resample_polyline(target_xy, 200)

    # ---- title -----------------------------------------------------------
    title = plot_title_11(
        order, source, target_dataset, target_name, target_demo, transport
    )

    # ---- plotting --------------------------------------------------------
    fig = None
    if live_plot:
        pos_init = state_init if order == "1st" else state_init[:, :2]
        fig = plt.figure(figsize=(8, 8))
        fig.show()
        plotter = TrajectoryPlotter(
            fig,
            x0=pos_init.T,
            pause_time=pause_time,
            goal=goal_tgt,
        )
        ax = plotter._ax
        if source_demo is not None:
            ax.plot(
                source_demo[:, 0],
                source_demo[:, 1],
                color="steelblue",
                linewidth=3,
                linestyle="--",
                alpha=0.6,
                label="Source demo",
                zorder=4,
            )
        ax.plot(
            target_demo_xy_resampled[:, 0],
            target_demo_xy_resampled[:, 1],
            color="lightgray",
            linewidth=6,
            label="Target demo",
            zorder=5,
        )
        ax.set_title(title)
        for t in range(1, visited_pos.shape[0]):
            plotter.update(visited_pos[t].T)
        plotter.draw()

    elif save_path:
        fig = plot_trajectories_demo_transport(
            visited_pos,
            goal_tgt,
            title,
            source_demo_xy=source_demo,
            target_demo_xy=target_demo_xy_resampled,
        )

    if save_path:
        if fig is None:
            fig = plot_trajectories_demo_transport(
                visited_pos,
                goal_tgt,
                title,
                source_demo_xy=source_demo,
                target_demo_xy=target_demo_xy_resampled,
            )
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if live_plot:
        plt.show()

    final = visited_pos[-1]
    pos_err = float(np.max(np.linalg.norm(final - goal_tgt, axis=1)))
    print(f"Source: {SOURCE_LABELS[source]}")
    print(
        f"Target: {target_dataset.upper()} '{target_name}' [demo {target_demo}]"
    )
    print(f"Transported attractor: {goal_tgt}, max final position error: {pos_err:.4e}")

    if order == "2nd":
        _, v_final = ds.split_state(visited[-1])
        vel_err = float(np.max(np.linalg.norm(v_final, axis=1)))
        print(f"  max final velocity norm: {vel_err:.4e}")

    return {
        "visited": visited,
        "visited_pos": visited_pos,
        "goal": goal_tgt,
        "source_demo": source_demo,
        "target_demo": target_demo_xy_resampled,
        "transport": transport,
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_demo_transport_cli_args(parser) -> None:
    """Register all ladder-11 CLI arguments on ``parser``."""
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="analytic",
        help="Source DS: analytic (ladder 1) or learned2/3/4 checkpoints.",
    )
    parser.add_argument(
        "--target-dataset",
        choices=TARGET_DATASET_CHOICES,
        default="lasa",
        metavar="DATASET",
        help="Target curve dataset family: 'lasa' or 'lair' (default: lasa).",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=_LASA_DEFAULT,
        metavar="NAME",
        help=(
            f"Dataset name within the chosen family.\n"
            f"  LASA examples: heee CShape Sine Snake Worm Sshape ...\n"
            f"  LAIR examples: capricorn e mountain phi double_loop ...\n"
            f"(default: {_LASA_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--target-demo",
        type=int,
        default=0,
        metavar="INDEX",
        help="Demonstration / episode index within the chosen dataset (default: 0).",
    )
    parser.add_argument(
        "--n-keypoints",
        type=int,
        default=40,
        metavar="N",
        help="Number of keypoints to resample both curves to before fitting (default: 40).",
    )
    parser.add_argument(
        "--grid-nx",
        type=int,
        default=4,
        help="Simulation grid columns in target demo bbox (default: 4, 12 points with --grid-ny 3).",
    )
    parser.add_argument(
        "--grid-ny",
        type=int,
        default=3,
        help="Simulation grid rows in target demo bbox (default: 3).",
    )
    parser.add_argument(
        "--grid-margin",
        type=float,
        default=0.12,
        metavar="FRAC",
        help="Inset fraction inside target demo bbox for initial positions (default: 0.12).",
    )
    parser.add_argument(
        "--no-affine-scale",
        action="store_true",
        help="Disable scale in the affine part of the fit (do_scale=False).",
    )
    parser.add_argument(
        "--no-affine-rotation",
        action="store_true",
        help="Disable rotation in the affine part of the fit (do_rotation=False).",
    )
    parser.add_argument(
        "--rbf-bandwidth",
        type=float,
        default=None,
        metavar="H",
        help="Gaussian RBF bandwidth h (default: auto via median heuristic).",
    )
    parser.add_argument(
        "--rbf-reg",
        type=float,
        default=1e-6,
        metavar="LAMBDA",
        help="Tikhonov regularisation for RBF solve (default: 1e-6).",
    )
    parser.add_argument(
        "--attractor",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("X", "Y"),
        help="Source-frame attractor for analytic source (default: 0 0).",
    )
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Analytic 1st-order gain k (source frame).")
    parser.add_argument("--kp", type=float, default=1.0,
                        help="Analytic 2nd-order position gain.")
    parser.add_argument(
        "--kd",
        type=float,
        default=None,
        help="Analytic 2nd-order velocity gain (default: 2*sqrt(kp)).",
    )
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Integration step for analytic source.")
    parser.add_argument(
        "--delta-t",
        type=float,
        default=0.1,
        help="PUMA transition step for learned sources (default: 0.1).",
    )
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of simulation steps (default: 2000).")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip interactive plotting (still saves if --save is set).",
    )
