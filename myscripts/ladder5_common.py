"""
Shared utilities for ladder step 5: DS transported under translation and rotation.

A source field ``f(x)`` defined in source coordinates is pushed to target coordinates
``y = R x + t`` via

    dy/dt = R f(R^T (y - t))          (1st order)

and the same pullback/pushforward for 2nd-order velocity and acceleration components.

Source systems (all prior ladder steps):

* ``analytic`` — straight-line DS from ladder 1
* ``learned2`` — PUMA checkpoint from ladder 2 (straight-line demo)
* ``learned3`` — PUMA checkpoint from ladder 3 (heee curved demo)
* ``learned4`` — PUMA checkpoint from ladder 4 (capricorn self-intersecting demo)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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

from ladder1a import AnalyticStraightLineDS, default_initial_states as default_positions_1st
from ladder1b import AnalyticStraightLineDS2ndOrder, default_initial_states as default_states_2nd

OrderTag = Literal["1st", "2nd"]
SourceTag = Literal["analytic", "learned2", "learned3", "learned4"]

SOURCE_CHOICES: tuple[str, ...] = ("analytic", "learned2", "learned3", "learned4")

SOURCE_LABELS: dict[SourceTag, str] = {
    "analytic": "analytic straight-line (ladder 1)",
    "learned2": "learned straight-line (ladder 2)",
    "learned3": "learned heee curved (ladder 3)",
    "learned4": "learned capricorn (ladder 4)",
}


@dataclass(frozen=True)
class RigidTransform2D:
    """Affine map y = R @ x + t with 2D rotation matrix R."""

    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_angle_and_translation(
        cls,
        angle_rad: float,
        translation: np.ndarray,
    ) -> RigidTransform2D:
        c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
        rotation = np.array([[c, -s], [s, c]], dtype=float)
        return cls(
            rotation=rotation,
            translation=np.asarray(translation, dtype=float).reshape(2),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (self.rotation @ x.T).T + self.translation

    def inverse(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return (self.rotation.T @ (y - self.translation).T).T

    def forward_vector(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return (self.rotation @ v.T).T

    def inverse_vector(self, v: np.ndarray) -> np.ndarray:
        return self.forward_vector(v)


def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    return RigidTransform2D.from_angle_and_translation(angle_rad, np.zeros(2)).rotation


def transform_positions(positions: np.ndarray, transform: RigidTransform2D) -> np.ndarray:
    return transform.forward(positions)


def transform_states_2nd(states: np.ndarray, transform: RigidTransform2D) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    q, v = states[:, :2], states[:, 2:]
    return np.hstack([transform.forward(q), transform.forward_vector(v)])


def default_initial_states_target(
    order: OrderTag,
    transform: RigidTransform2D,
    *,
    source: SourceTag = "analytic",
    data: dict | None = None,
) -> np.ndarray:
    """
    Initial states in the transported (target) frame.

    Uses the ladder-1 position grid for ``analytic`` / ``learned2``; for curved demos
    uses a workspace grid in source frame then maps it forward.
    """
    if source in ("learned3", "learned4") and data is not None:
        from ladder3_common import initial_states_for_order as init_l3
        from ladder4_common import initial_states_for_order as init_l4

        init_fn = init_l3 if source == "learned3" else init_l4
        states_src = init_fn(order, data)
        if order == "1st":
            return transform.forward(states_src)
        return transform_states_2nd(states_src, transform)

    if order == "1st":
        return transform.forward(default_positions_1st())
    return transform_states_2nd(default_states_2nd(), transform)


def _load_learned_source(source: SourceTag, order: OrderTag):
    if source == "learned2":
        from ladder2_common import load_learned_ds

        return load_learned_ds(order, verbose=False)
    if source == "learned3":
        from ladder3_common import load_learned_ds

        return load_learned_ds(order, verbose=False)
    from ladder4_common import load_learned_ds

    return load_learned_ds(order, verbose=False)


def _demonstration_xy(source: SourceTag, data: dict | None) -> np.ndarray | None:
    if source == "analytic":
        return None
    if source == "learned2":
        from ladder2_data import load_demonstration_xy

        return load_demonstration_xy()
    if source == "learned3":
        from ladder3_common import demonstration_xy_from_data

        return demonstration_xy_from_data(data)
    from ladder4_common import demonstration_xy_from_data

    return demonstration_xy_from_data(data)


def _denormalize_positions(pos_norm: np.ndarray, data: dict) -> np.ndarray:
    from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import denormalize_state

    x_min = np.asarray(data["x min"]).reshape(-1)
    x_max = np.asarray(data["x max"]).reshape(-1)
    flat = np.asarray(pos_norm, dtype=float).reshape(-1, pos_norm.shape[-1])
    flat = denormalize_state(flat, x_min, x_max)
    return flat.reshape(pos_norm.shape)


def _denormalize_velocities(vel_norm: np.ndarray, dynamical_system) -> np.ndarray:
    from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import denormalize_state

    min_vel = dynamical_system.min_vel.detach().cpu().numpy()
    max_vel = dynamical_system.max_vel.detach().cpu().numpy()
    flat = np.asarray(vel_norm, dtype=float).reshape(-1, vel_norm.shape[-1])
    flat = denormalize_state(flat, min_vel, max_vel)
    return flat.reshape(vel_norm.shape)


def simulate_learned_then_transport(
    learner,
    data: dict,
    order: OrderTag,
    transform: RigidTransform2D,
    init_target: np.ndarray,
    n_steps: int,
    *,
    delta_t: float = 0.1,
) -> np.ndarray:
    """
    Roll out a learned source DS in demonstration coordinates, then map to target frame.

    Uses PUMA ``transition`` steps of length ``delta_t`` (same as ladder 2–4), avoiding
    Euler integration with a mismatched ``dt``.
    """
    import torch
    from pumafabrics.puma_adapted.agent.neural_network import DEVICE
    from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import normalize_state

    init_target = np.asarray(init_target, dtype=float)
    if init_target.ndim == 1:
        init_target = init_target.reshape(1, -1)

    dim = 2
    n_traj = init_target.shape[0]
    x_min = data["x min"]
    x_max = data["x max"]

    if order == "1st":
        pos_src = transform.inverse(init_target)
        state_norm = normalize_state(pos_src, x_min, x_max)
        state_dim = dim
    else:
        q_tgt, v_tgt = init_target[:, :dim], init_target[:, dim:]
        q_src = transform.inverse(q_tgt)
        v_src = transform.inverse_vector(v_tgt)
        q_norm = normalize_state(q_src, x_min, x_max)
        state_norm = np.hstack([q_norm, v_src])
        state_dim = 2 * dim

    x_gpu = torch.FloatTensor(state_norm).to(DEVICE)
    dynamical_system = learner.init_dynamical_system(initial_states=x_gpu, delta_t=delta_t)

    visited = np.zeros((n_steps + 1, n_traj, state_dim))
    visited[0] = init_target

    x_t = x_gpu
    for t in range(1, n_steps + 1):
        with torch.no_grad():
            x_t = dynamical_system.transition(space="task", x_t=x_t)["desired state"]
        x_np = x_t.detach().cpu().numpy()
        if order == "1st":
            pos_src = _denormalize_positions(x_np, data)
            visited[t] = transform.forward(pos_src)
        else:
            pos_src = _denormalize_positions(x_np[:, :dim], data)
            vel_src = _denormalize_velocities(x_np[:, dim:], dynamical_system)
            q_tgt = transform.forward(pos_src)
            v_tgt = transform.forward_vector(vel_src)
            visited[t] = np.hstack([q_tgt, v_tgt])

    return visited


def learned_derivative_in_source_frame(
    learner,
    data: dict,
    order: OrderTag,
    state_src: np.ndarray,
    *,
    delta_t: float = 0.1,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the learned field in source (demonstration) coordinates.

    Returns task-space velocity (1st order) or (velocity, acceleration) for 2nd order.
    """
    import torch
    from pumafabrics.puma_adapted.agent.neural_network import DEVICE
    from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import normalize_state

    state_src = np.asarray(state_src, dtype=float)
    if state_src.ndim == 1:
        state_src = state_src.reshape(1, -1)

    x_min = data["x min"]
    x_max = data["x max"]

    if order == "1st":
        state_norm = normalize_state(state_src, x_min, x_max)
    else:
        q, v = state_src[:, :2], state_src[:, 2:]
        q_norm = normalize_state(q, x_min, x_max)
        state_norm = np.hstack([q_norm, v])

    x_gpu = torch.FloatTensor(state_norm).to(DEVICE)
    dynamical_system = learner.init_dynamical_system(initial_states=x_gpu, delta_t=delta_t)
    with torch.no_grad():
        info = dynamical_system.transition(space="task", x_t=x_gpu)

    if order == "1st":
        return info["desired velocity"].detach().cpu().numpy()

    acc = info["desired acceleration"]
    if acc is None:
        raise RuntimeError("2nd-order learned DS did not return acceleration.")
    vel = info["desired velocity"].detach().cpu().numpy()[:, :2]
    return vel, acc.detach().cpu().numpy()[:, :2]


class TransportedFirstOrderDS:
    """1st-order DS transported from a source field under rigid transform."""

    def __init__(
        self,
        source: SourceTag,
        transform: RigidTransform2D,
        *,
        attractor: np.ndarray | None = None,
        gain: float = 1.0,
        learner=None,
        data: dict | None = None,
        dt: float = 0.01,
        delta_t: float = 0.1,
    ):
        self.source = source
        self.transform = transform
        self.dt = float(dt)
        self.delta_t = float(delta_t)
        self.dim = 2
        self.learner = learner
        self.data = data

        if source == "analytic":
            self._analytic = AnalyticStraightLineDS(
                attractor=np.asarray(attractor if attractor is not None else [0.0, 0.0]),
                gain=gain,
                dt=dt,
            )
        else:
            if learner is None or data is None:
                raise ValueError("learned source requires learner and data.")

    @property
    def attractor_target(self) -> np.ndarray:
        if self.source == "analytic":
            return self.transform.forward(self._analytic.attractor)
        goal = np.asarray(self.data["goals"][0], dtype=float).reshape(2)
        return self.transform.forward(goal)

    def velocity(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        x = self.transform.inverse(y)
        if self.source == "analytic":
            v_src = self._analytic.velocity(x)
        else:
            v_src = learned_derivative_in_source_frame(
                self.learner,
                self.data,
                "1st",
                x,
                delta_t=self.delta_t,
            )
        return self.transform.forward_vector(v_src)

    def step(self, y: np.ndarray) -> np.ndarray:
        return y + self.velocity(y) * self.dt

    def simulate(self, y_init: np.ndarray, n_steps: int) -> np.ndarray:
        y_init = np.asarray(y_init, dtype=float)
        if y_init.ndim == 1:
            y_init = y_init.reshape(1, -1)
        if self.source != "analytic":
            return simulate_learned_then_transport(
                self.learner,
                self.data,
                "1st",
                self.transform,
                y_init,
                n_steps,
                delta_t=self.delta_t,
            )
        n_traj = y_init.shape[0]
        visited = np.zeros((n_steps + 1, n_traj, self.dim))
        visited[0] = y_init
        y = y_init.copy()
        for t in range(1, n_steps + 1):
            y = self.step(y)
            visited[t] = y
        return visited


class TransportedSecondOrderDS:
    """2nd-order DS transported from a source field under rigid transform."""

    def __init__(
        self,
        source: SourceTag,
        transform: RigidTransform2D,
        *,
        attractor: np.ndarray | None = None,
        kp: float = 1.0,
        kd: float | None = None,
        learner=None,
        data: dict | None = None,
        dt: float = 0.01,
        delta_t: float = 0.1,
    ):
        self.source = source
        self.transform = transform
        self.dt = float(dt)
        self.delta_t = float(delta_t)
        self.dim = 2
        self.state_dim = 4
        self.learner = learner
        self.data = data

        if source == "analytic":
            self._analytic = AnalyticStraightLineDS2ndOrder(
                attractor=np.asarray(attractor if attractor is not None else [0.0, 0.0]),
                kp=kp,
                kd=kd,
                dt=dt,
            )
        else:
            if learner is None or data is None:
                raise ValueError("learned source requires learner and data.")

    @property
    def attractor_target(self) -> np.ndarray:
        if self.source == "analytic":
            return self.transform.forward(self._analytic.attractor)
        goal = np.asarray(self.data["goals"][0], dtype=float).reshape(2)
        return self.transform.forward(goal)

    def split_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)
        return state[..., : self.dim], state[..., self.dim :]

    def stack_state(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.hstack([q, v])

    def acceleration(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        x = self.transform.inverse(q)
        v_src = self.transform.inverse_vector(v)
        if self.source == "analytic":
            a_src = self._analytic.acceleration(x, v_src)
        else:
            _, a_src = learned_derivative_in_source_frame(
                self.learner,
                self.data,
                "2nd",
                self.stack_state(x, v_src),
                delta_t=self.delta_t,
            )
        return self.transform.forward_vector(a_src)

    def step(self, state: np.ndarray) -> np.ndarray:
        q, v = self.split_state(state)
        a = self.acceleration(q, v)
        q_next = q + v * self.dt
        v_next = v + a * self.dt
        return self.stack_state(q_next, v_next)

    def simulate(self, state_init: np.ndarray, n_steps: int) -> np.ndarray:
        state_init = np.asarray(state_init, dtype=float)
        if state_init.ndim == 1:
            state_init = state_init.reshape(1, -1)
        if self.source != "analytic":
            return simulate_learned_then_transport(
                self.learner,
                self.data,
                "2nd",
                self.transform,
                state_init,
                n_steps,
                delta_t=self.delta_t,
            )
        n_traj = state_init.shape[0]
        visited = np.zeros((n_steps + 1, n_traj, self.state_dim))
        visited[0] = state_init
        state = state_init.copy()
        for t in range(1, n_steps + 1):
            state = self.step(state)
            visited[t] = state
        return visited

    def positions(self, visited: np.ndarray) -> np.ndarray:
        return visited[..., : self.dim]


def build_transported_ds(
    order: OrderTag,
    source: SourceTag,
    transform: RigidTransform2D,
    *,
    attractor: np.ndarray,
    gain: float = 1.0,
    kp: float = 1.0,
    kd: float | None = None,
    dt: float = 0.01,
    delta_t: float = 0.1,
):
    learner, data = None, None
    if source != "analytic":
        learner, _, data, _ = _load_learned_source(source, order)

    if order == "1st":
        return TransportedFirstOrderDS(
            source,
            transform,
            attractor=attractor,
            gain=gain,
            learner=learner,
            data=data,
            dt=dt,
            delta_t=delta_t,
        ), data
    return TransportedSecondOrderDS(
        source,
        transform,
        attractor=attractor,
        kp=kp,
        kd=kd,
        learner=learner,
        data=data,
        dt=dt,
        delta_t=delta_t,
    ), data


def plot_title(order: OrderTag, source: SourceTag, angle_deg: float) -> str:
    order_label = "1st-order" if order == "1st" else "2nd-order"
    return (
        f"Transported {order_label} DS ({SOURCE_LABELS[source]})\n"
        f"rotation {angle_deg:.1f}°"
    )


def plot_trajectories_static(
    visited_pos: np.ndarray,
    attractor: np.ndarray,
    title: str,
    *,
    demonstration_xy: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
    else:
        fig = ax.figure

    if demonstration_xy is not None:
        ax.plot(
            demonstration_xy[:, 0],
            demonstration_xy[:, 1],
            color="lightgray",
            linewidth=6,
            label="Demonstration (transported)",
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


def run_transport_pipeline(
    order: OrderTag,
    *,
    source: SourceTag,
    translation: np.ndarray,
    angle_rad: float,
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
    transform = RigidTransform2D.from_angle_and_translation(angle_rad, translation)
    ds, data = build_transported_ds(
        order,
        source,
        transform,
        attractor=attractor,
        gain=gain,
        kp=kp,
        kd=kd,
        dt=dt,
        delta_t=delta_t,
    )

    state_init = default_initial_states_target(order, transform, source=source, data=data)
    if order == "1st":
        visited = ds.simulate(state_init, n_steps)
        visited_pos = visited
    else:
        visited = ds.simulate(state_init, n_steps)
        visited_pos = ds.positions(visited)

    goal_tgt = ds.attractor_target
    demo_src = _demonstration_xy(source, data)
    demo_tgt = transform.forward(demo_src) if demo_src is not None else None

    angle_deg = float(np.degrees(angle_rad))
    title = plot_title(order, source, angle_deg)

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
        if demo_tgt is not None:
            plotter._ax.plot(
                demo_tgt[:, 0],
                demo_tgt[:, 1],
                color="lightgray",
                linewidth=6,
                zorder=5,
            )
        plotter._ax.set_title(title)
        for t in range(1, visited_pos.shape[0]):
            plotter.update(visited_pos[t].T)
        plotter.draw()
    elif save_path:
        fig = plot_trajectories_static(
            visited_pos,
            goal_tgt,
            title,
            demonstration_xy=demo_tgt,
        )

    if save_path:
        if fig is None:
            fig = plot_trajectories_static(
                visited_pos,
                goal_tgt,
                title,
                demonstration_xy=demo_tgt,
            )
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if live_plot:
        plt.show()

    final = visited_pos[-1]
    pos_err = np.max(np.linalg.norm(final - goal_tgt, axis=1))
    print(f"Source: {SOURCE_LABELS[source]}")
    print(f"Transform: translation={translation}, angle={angle_deg:.2f}°")
    print(f"Transported attractor: {goal_tgt}, max final position error: {pos_err:.4e}")

    if order == "2nd":
        q_final, v_final = ds.split_state(visited[-1])
        vel_err = np.max(np.linalg.norm(v_final, axis=1))
        print(f"  max final velocity norm: {vel_err:.4e}")

    return {
        "visited": visited,
        "visited_pos": visited_pos,
        "goal": goal_tgt,
        "demonstration": demo_tgt,
        "transform": transform,
    }


def add_transport_cli_args(parser) -> None:
    """Register ladder-5 CLI arguments on an ``ArgumentParser``."""
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="analytic",
        help="Source DS: analytic (ladder 1) or learned2/3/4 checkpoints.",
    )
    parser.add_argument(
        "--translation",
        type=float,
        nargs=2,
        default=[1.0, 0.5],
        metavar=("TX", "TY"),
        help="Translation t in target frame (default: 1.0 0.5).",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=45.0,
        help="Rotation angle in degrees (default: 45).",
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
                        help="Integration step size.")
    parser.add_argument(
        "--delta-t",
        type=float,
        default=0.1,
        help="PUMA transition step for learned sources (one sim step = delta-t; same as ladder 2–4).",
    )
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of simulation steps.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip interactive plotting (still saves if --save is set).",
    )
