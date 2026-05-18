"""
Ladder step 1b: analytic 2nd-order straight-line dynamical system with a known attractor.

    q_dot = v
    v_dot = -k_p (q - q*) - k_d v

State is stacked [q, v]. With zero initial velocity, trajectories in task space are
straight lines converging to q* with v -> 0.
"""

from __future__ import annotations
from pumafabrics.puma_adapted.tools.animation import TrajectoryPlotter

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class AnalyticStraightLineDS2ndOrder:
    """Second-order linear DS with a single known position attractor."""

    def __init__(
        self,
        attractor: np.ndarray,
        kp: float = 1.0,
        kd: float | None = None,
        dt: float = 0.01,
    ):
        self.attractor = np.asarray(attractor, dtype=float).reshape(-1)
        self.kp = float(kp)
        self.kd = float(2.0 * np.sqrt(kp) if kd is None else kd)
        self.dt = float(dt)
        self.dim = self.attractor.size
        self.state_dim = 2 * self.dim

    def split_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)
        q = state[..., : self.dim]
        v = state[..., self.dim:]
        return q, v

    def stack_state(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.concatenate([q, v], axis=-1)

    def acceleration(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return -self.kp * (q - self.attractor) - self.kd * v

    def step(self, state: np.ndarray) -> np.ndarray:
        q, v = self.split_state(state)
        a = self.acceleration(q, v)
        q_next = q + v * self.dt
        v_next = v + a * self.dt
        return self.stack_state(q_next, v_next)

    def simulate(self, state_init: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Roll out the DS from batch initial states.

        Parameters
        ----------
        state_init : (n_trajectories, 2 * dim)  — [q, v] per row
        n_steps : int

        Returns
        -------
        visited : (n_steps + 1, n_trajectories, 2 * dim)
        """
        state_init = np.asarray(state_init, dtype=float)
        if state_init.ndim == 1:
            state_init = state_init.reshape(1, -1)
        if state_init.shape[1] != self.state_dim:
            raise ValueError(
                f"state_init has dim {state_init.shape[1]}, expected {self.state_dim}",
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
        """Extract position trajectories from full state history."""
        return visited[..., : self.dim]


def default_initial_states(zero_velocity: bool = True) -> np.ndarray:
    """2D positions + velocities (same positions as ladder1a / simulate_ds_trial)."""
    positions = np.array(
        [
            [0.5, 0.6],
            [-0.75, 0.9],
            [0.9, -0.9],
            [-0.9, -0.9],
            [0.9, 0.9],
            [0.9, 0.3],
            [-0.9, -0.1],
            [-0.9, 0.0],
            [0.4, 0.4],
            [0.9, -0.1],
            [-0.9, -0.5],
            [0.9, -0.5],
        ],
        dtype=float,
    )
    if zero_velocity:
        velocities = np.zeros_like(positions)
    else:
        velocities = 0.1 * (positions - positions.mean(axis=0))
    return np.hstack([positions, velocities])


def plot_trajectories_static(
    visited_positions: np.ndarray,
    attractor: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
    else:
        fig = ax.figure

    n_traj = visited_positions.shape[1]
    cmap = plt.get_cmap("gist_rainbow")
    for i in range(n_traj):
        color = cmap(i / max(n_traj - 1, 1))
        ax.plot(
            visited_positions[:, i, 0],
            visited_positions[:, i, 1],
            color=color,
            linewidth=2.0,
        )
    ax.plot(attractor[0], attractor[1], "g*", markersize=15, label="Attractor")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Analytic 2nd-order straight-line DS")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    return fig


def run_simulation(
    attractor: np.ndarray,
    state_init: np.ndarray,
    n_steps: int = 2000,
    kp: float = 1.0,
    kd: float | None = None,
    dt: float = 0.01,
    live_plot: bool = True,
    pause_time: float = 1e-5,
    save_path: str | None = None,
) -> np.ndarray:
    ds = AnalyticStraightLineDS2ndOrder(
        attractor=attractor, kp=kp, kd=kd, dt=dt)
    visited = ds.simulate(state_init, n_steps)
    visited_pos = ds.positions(visited)
    pos_init = state_init[:, : ds.dim]

    fig = None
    if live_plot:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.show()
        plotter = TrajectoryPlotter(
            fig,
            x0=pos_init.T,
            pause_time=pause_time,
            goal=attractor,
        )
        for t in range(visited_pos.shape[0]):
            plotter.update(visited_pos[t].T)
    elif save_path:
        fig = plot_trajectories_static(visited_pos, attractor)

    if save_path:
        if fig is None:
            fig = plot_trajectories_static(visited_pos, attractor)
        os.makedirs(os.path.dirname(os.path.abspath(save_path))
                    or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if live_plot:
        plt.show()

    return visited


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ladder 1b: analytic 2nd-order straight-line DS with known attractor.",
    )
    parser.add_argument(
        "--attractor",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("X", "Y"),
        help="Attractor position q* (default: 0 0).",
    )
    parser.add_argument("--kp", type=float, default=1.0,
                        help="Position gain k_p.")
    parser.add_argument(
        "--kd",
        type=float,
        default=None,
        help="Velocity gain k_d (default: 2*sqrt(k_p), critical damping).",
    )
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Integration step size.")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of simulation steps.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip interactive plotting (still saves if --save is set).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join(os.path.dirname(__file__),
                             "images", "ladder1b_analytic_ds_2nd_order.png"),
        help="Path for output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attractor = np.array(args.attractor, dtype=float)
    state_init = default_initial_states()

    visited = run_simulation(
        attractor=attractor,
        state_init=state_init,
        n_steps=args.steps,
        kp=args.kp,
        kd=args.kd,
        dt=args.dt,
        live_plot=not args.no_plot,
        save_path=args.save,
    )

    ds = AnalyticStraightLineDS2ndOrder(
        attractor=attractor, kp=args.kp, kd=args.kd, dt=args.dt)
    q_final, v_final = ds.split_state(visited[-1])
    pos_err = np.max(np.linalg.norm(q_final - attractor, axis=1))
    vel_err = np.max(np.linalg.norm(v_final, axis=1))
    print(
        f"Attractor: {attractor}, kp={args.kp}, kd={ds.kd:.4f}\n"
        f"  max final position error: {pos_err:.4e}\n"
        f"  max final velocity norm:  {vel_err:.4e}",
    )


if __name__ == "__main__":
    main()
