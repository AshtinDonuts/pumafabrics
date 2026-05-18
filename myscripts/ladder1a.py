"""
Ladder step 1: analytic straight-line dynamical system with a known attractor.

    dx/dt = -k (x - x*)

Trajectories are straight lines in task space converging to x*.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow running as: python myscripts/ladder1a.py from repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pumafabrics.puma_adapted.tools.animation import TrajectoryPlotter


class AnalyticStraightLineDS:
    """First-order linear DS with a single known attractor."""

    def __init__(self, attractor: np.ndarray, gain: float = 1.0, dt: float = 0.01):
        self.attractor = np.asarray(attractor, dtype=float).reshape(-1)
        self.gain = float(gain)
        self.dt = float(dt)
        self.dim = self.attractor.size

    def velocity(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return -self.gain * (x - self.attractor)

    def step(self, x: np.ndarray) -> np.ndarray:
        return x + self.velocity(x) * self.dt

    def simulate(self, x_init: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Roll out the DS from batch initial positions.

        Parameters
        ----------
        x_init : (n_trajectories, dim)
        n_steps : int

        Returns
        -------
        visited : (n_steps + 1, n_trajectories, dim)
        """
        x_init = np.asarray(x_init, dtype=float)
        if x_init.ndim == 1:
            x_init = x_init.reshape(1, -1)
        if x_init.shape[1] != self.dim:
            raise ValueError(f"x_init has dim {x_init.shape[1]}, expected {self.dim}")

        n_traj = x_init.shape[0]
        visited = np.zeros((n_steps + 1, n_traj, self.dim))
        visited[0] = x_init
        x = x_init.copy()
        for t in range(1, n_steps + 1):
            x = self.step(x)
            visited[t] = x
        return visited


def default_initial_states() -> np.ndarray:
    """Grid of 2D positions (same layout as simulate_ds_trial, position only)."""
    return np.array(
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


def plot_trajectories_static(
    visited: np.ndarray,
    attractor: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot all rolled-out trajectories and the attractor."""
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
    else:
        fig = ax.figure

    n_traj = visited.shape[1]
    cmap = plt.get_cmap("gist_rainbow")
    for i in range(n_traj):
        color = cmap(i / max(n_traj - 1, 1))
        ax.plot(visited[:, i, 0], visited[:, i, 1], color=color, linewidth=2.0)
    ax.plot(attractor[0], attractor[1], "g*", markersize=15, label="Attractor")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Analytic straight-line DS")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    return fig


def run_simulation(
    attractor: np.ndarray,
    x_init: np.ndarray,
    n_steps: int = 2000,
    gain: float = 1.0,
    dt: float = 0.01,
    live_plot: bool = True,
    pause_time: float = 1e-5,
    save_path: str | None = None,
) -> np.ndarray:
    ds = AnalyticStraightLineDS(attractor=attractor, gain=gain, dt=dt)
    visited = ds.simulate(x_init, n_steps)

    fig = None
    if live_plot:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.show()
        plotter = TrajectoryPlotter(
            fig,
            x0=x_init.T,
            pause_time=pause_time,
            goal=attractor,
        )
        for t in range(visited.shape[0]):
            plotter.update(visited[t].T)
    elif save_path:
        fig = plot_trajectories_static(visited, attractor)

    if save_path:
        if fig is None:
            fig = plot_trajectories_static(visited, attractor)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if live_plot:
        plt.show()

    return visited


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ladder 1: analytic straight-line DS with known attractor.",
    )
    parser.add_argument(
        "--attractor",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("X", "Y"),
        help="Attractor position x* (default: 0 0).",
    )
    parser.add_argument("--gain", type=float, default=1.0, help="Convergence rate k.")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration step size.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of simulation steps.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip interactive plotting (still saves if --save is set).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "images", "ladder1a_analytic_ds.png"),
        help="Path for output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attractor = np.array(args.attractor, dtype=float)
    x_init = default_initial_states()

    visited = run_simulation(
        attractor=attractor,
        x_init=x_init,
        n_steps=args.steps,
        gain=args.gain,
        dt=args.dt,
        live_plot=not args.no_plot,
        save_path=args.save,
    )

    final = visited[-1]
    max_err = np.max(np.linalg.norm(final - attractor, axis=1))
    print(f"Attractor: {attractor}, max final error: {max_err:.4e}")


if __name__ == "__main__":
    main()
