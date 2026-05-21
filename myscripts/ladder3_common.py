"""
Shared utilities for ladder step 3: learned DS from one curved heee demonstration.

Training delegates to ``pumafabrics.puma_adapted.train``; this module handles dataset
setup, checkpoint policy, grid simulation, and plotting.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pumafabrics.puma_adapted.agent.neural_network import DEVICE
from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import denormalize_state
from pumafabrics.puma_adapted.initializer import initialize_framework
from pumafabrics.puma_adapted.tools.animation import TrajectoryPlotter
from pumafabrics.puma_adapted.train import configure_params, run_training

_MYSCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _MYSCRIPTS not in sys.path:
    sys.path.insert(0, _MYSCRIPTS)

from ladder3_data import ensure_heee_dataset

OrderTag = Literal["1st", "2nd"]

PARAMS_BY_ORDER: dict[OrderTag, str] = {
    "1st": "ladder3_1st_order_2D",
    "2nd": "ladder3_2nd_order_2D",
}

PLOT_TITLE_BY_ORDER: dict[OrderTag, str] = {
    "1st": "Learned 1st-order DS (heee curved demo)",
    "2nd": "Learned 2nd-order DS (heee curved demo)",
}


def repo_results_base() -> str:
    return _REPO_ROOT + os.sep


def model_checkpoint_path(params) -> str:
    """Checkpoint path (params.results_path already ends with ``<primitive_id>/``)."""
    return os.path.join(params.results_path, "model")


def build_params(order: OrderTag, load_model: bool = False, train: bool = True):
    params, params_name = configure_params(
        PARAMS_BY_ORDER[order],
        repo_results_base(),
    )
    params.load_model = load_model
    params.train = train
    return params, params_name


def tensorboard_log_dir(params_name: str, selected_primitives_ids: str) -> str:
    log_name = f"{params_name}_{selected_primitives_ids}"
    return os.path.join(repo_results_base(), "results", "tensorboard_runs", log_name)


def read_last_training_iteration(results_path: str) -> int | None:
    """
    Last completed evaluation iteration from ``training_evaluation_summary.txt``,
    or the highest ``primitive_*_iter_*.pdf`` index under ``images/``.
    """
    last: int | None = None
    summary = os.path.join(results_path, "training_evaluation_summary.txt")
    if os.path.isfile(summary):
        with open(summary, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Iteration number:"):
                    last = int(line.split(":", 1)[1].strip())

    images_dir = os.path.join(results_path, "images")
    if os.path.isdir(images_dir):
        pat = re.compile(r"primitive_\d+_iter_(\d+)\.")
        for name in os.listdir(images_dir):
            match = pat.search(name)
            if match:
                it = int(match.group(1))
                last = it if last is None else max(last, it)
    return last


def resolve_n_simulate(n_simulate: int | None, params) -> int | None:
    """
    Interval for ladder grid plots during training.

    * ``None`` -> ``params.evaluation_interval`` (same cadence as ``primitive_*_iter_*.pdf``)
    * ``0`` -> disabled
    * ``N > 0`` -> every N iterations
    """
    if n_simulate is None:
        return int(params.evaluation_interval)
    if n_simulate <= 0:
        return None
    return n_simulate


def _periodic_simulate_plot_path(results_path: str, iteration: int) -> str:
    images_dir = os.path.join(results_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    return os.path.join(images_dir, f"ladder_sim_iter_{iteration:06d}.png")


def make_periodic_simulate_callback(
    order: OrderTag,
    results_path: str,
    n_steps: int,
) -> Callable[[int, object, object, dict], None]:
    """Build a training callback that runs ladder grid sim + static PNG."""

    def callback(iteration: int, learner, _evaluator, data: dict) -> None:
        out_path = _periodic_simulate_plot_path(results_path, iteration)
        print(f"Ladder grid simulation at iteration {iteration} -> {out_path}")
        was_training = learner.model.training
        learner.model.eval()
        try:
            demo_xy = demonstration_xy_from_data(data)
            goal = np.asarray(data["goals"][0], dtype=float).reshape(-1)
            with torch.no_grad():
                visited = simulate_learned_ds(learner, data, order, n_steps=n_steps)
            title = f"{PLOT_TITLE_BY_ORDER[order]} (iter {iteration})"
            fig = plot_learned_ds_static(visited, demo_xy, goal, title)
            fig.savefig(out_path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            print(f"Saved ladder simulation plot to {out_path}")
        except Exception:
            plt.close("all")
            raise
        finally:
            learner.model.train(was_training)

    return callback


def save_final_checkpoint(learner, ckpt: str, verbose: bool = True) -> None:
    """
    Persist weights after a full training run.

    ``train.run_training`` only saves via evaluator on best metric; this guarantees a
    loadable ``model`` file for ``--simulate-only``.
    """
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save(learner.model.state_dict(), ckpt)
    if verbose:
        print(f"Saved model to {ckpt}")


def _run_training_with_optional_simulate(
    order: OrderTag,
    params,
    params_name: str,
    *,
    start_iteration: int = 0,
    n_simulate: int | None = None,
    n_steps: int = 2000,
    verbose: bool = True,
) -> tuple[object, object, dict, float]:
    interval = resolve_n_simulate(n_simulate, params)
    step_callback = None
    if interval is not None:
        if verbose:
            print(f"Ladder grid plots every {interval} training iterations.")
        step_callback = make_periodic_simulate_callback(order, params.results_path, n_steps)
    return run_training(
        params,
        params_name,
        tensorboard_log_dir=tensorboard_log_dir(params_name, params.selected_primitives_ids),
        start_iteration=start_iteration,
        step_callback=step_callback,
        step_callback_interval=interval,
        verbose=verbose,
    )


def train_learned_ds(
    order: OrderTag,
    max_iterations: int | None = None,
    force: bool = False,
    n_simulate: int | None = None,
    n_steps: int = 2000,
    verbose: bool = True,
) -> tuple[object, object, dict, str]:
    """
    Train (or load) a PUMA model on the single curved heee demonstration.

    Returns learner, evaluator, data, params_module_name.
    """
    ensure_heee_dataset()
    params, params_name = build_params(order, load_model=False, train=True)
    if max_iterations is not None:
        params.max_iterations = max_iterations

    ckpt = model_checkpoint_path(params)
    last_iter = read_last_training_iteration(params.results_path) if os.path.isfile(ckpt) else None

    if os.path.isfile(ckpt) and not force:
        if last_iter is not None and last_iter < params.max_iterations:
            start_iteration = last_iter + 1
            if verbose:
                print(
                    f"Checkpoint found; resuming training from iteration "
                    f"{start_iteration} to {params.max_iterations}.",
                )
            params.load_model = True
            learner, evaluator, data, _ = _run_training_with_optional_simulate(
                order,
                params,
                params_name,
                start_iteration=start_iteration,
                n_simulate=n_simulate,
                n_steps=n_steps,
                verbose=verbose,
            )
            save_final_checkpoint(learner, ckpt, verbose=verbose)
            return learner, evaluator, data, params_name

        if verbose:
            print(f"Checkpoint exists at {ckpt}; no further training needed.")
            if last_iter is not None:
                print(
                    f"Last recorded iteration is {last_iter} "
                    f"(target {params.max_iterations}).",
                )
            print("Use --force-train to restart from scratch.")
        return load_learned_ds(order, verbose=verbose)

    learner, evaluator, data, _ = _run_training_with_optional_simulate(
        order,
        params,
        params_name,
        n_simulate=n_simulate,
        n_steps=n_steps,
        verbose=verbose,
    )
    save_final_checkpoint(learner, ckpt, verbose=verbose)
    return learner, evaluator, data, params_name


def load_learned_ds(order: OrderTag, verbose: bool = False):
    ensure_heee_dataset()
    params, params_name = build_params(order, load_model=True, train=False)
    ckpt = model_checkpoint_path(params)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"No trained model at {ckpt}. Run with --train or omit --simulate-only.",
        )
    learner, evaluator, data = initialize_framework(params, params_name, verbose=verbose)
    return learner, evaluator, data, params_name


def workspace_initial_positions(
    x_min: np.ndarray,
    x_max: np.ndarray,
    *,
    nx: int = 4,
    ny: int = 3,
    margin: float = 0.12,
) -> np.ndarray:
    """
    Grid of 2D positions inset within PUMA workspace bounds (same count as ladder1a).
    """
    x_min = np.asarray(x_min, dtype=float).reshape(-1)
    x_max = np.asarray(x_max, dtype=float).reshape(-1)
    span = x_max - x_min
    lo = x_min + margin * span
    hi = x_max - margin * span
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def initial_states_for_order(order: OrderTag, data: dict) -> np.ndarray:
    """Initial states spanning the demonstration workspace (from training bounds)."""
    positions = workspace_initial_positions(data["x min"], data["x max"])
    if order == "1st":
        return positions
    velocities = np.zeros_like(positions)
    return np.hstack([positions, velocities])


def demonstration_xy_from_data(data: dict, demo_index: int = 0) -> np.ndarray:
    """Demonstration positions in task space (T, 2), same frame as simulation."""
    raw = np.asarray(data["demonstrations raw"][demo_index], dtype=float)
    if raw.ndim == 2 and raw.shape[0] == 2:
        return raw.T
    return np.asarray(raw, dtype=float).reshape(-1, 2)


def _axis_limits(
    demo_xy: np.ndarray,
    visited_pos: np.ndarray,
    *,
    buffer: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = np.concatenate([demo_xy[:, 0], visited_pos[:, :, 0].ravel()])
    ys = np.concatenate([demo_xy[:, 1], visited_pos[:, :, 1].ravel()])
    x_span = xs.max() - xs.min()
    y_span = ys.max() - ys.min()
    pad_x = buffer * (x_span if x_span > 0 else 1.0)
    pad_y = buffer * (y_span if y_span > 0 else 1.0)
    return (xs.min() - pad_x, xs.max() + pad_x), (ys.min() - pad_y, ys.max() + pad_y)


def _denormalize_positions(pos_norm: np.ndarray, data: dict) -> np.ndarray:
    x_min = np.asarray(data["x min"]).reshape(-1)
    x_max = np.asarray(data["x max"]).reshape(-1)
    flat = pos_norm.reshape(-1, pos_norm.shape[-1])
    flat = denormalize_state(flat, x_min, x_max)
    return flat.reshape(pos_norm.shape)


def simulate_learned_ds(
    learner,
    data: dict,
    order: OrderTag,
    n_steps: int = 2000,
    delta_t: float = 0.1,
) -> np.ndarray:
    """Roll out learned DS; returns denormalized position history (T+1, n_traj, 2)."""
    state_init = initial_states_for_order(order, data)
    x_min = data["x min"]
    x_max = data["x max"]
    from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import normalize_state

    if order == "1st":
        pos_init = state_init
        state_norm = normalize_state(pos_init, x_min, x_max)
    else:
        pos_init = state_init[:, :2]
        vel_init = state_init[:, 2:]
        pos_norm = normalize_state(pos_init, x_min, x_max)
        state_norm = np.hstack([pos_norm, vel_init])

    x_gpu = torch.FloatTensor(state_norm).to(DEVICE)
    dynamical_system = learner.init_dynamical_system(initial_states=x_gpu, delta_t=delta_t)

    dim = 2
    n_traj = state_init.shape[0]
    visited_norm = np.zeros((n_steps + 1, n_traj, dim))
    visited_norm[0] = state_norm[:, :dim] if order == "2nd" else state_norm

    x_t = x_gpu
    for t in range(1, n_steps + 1):
        x_t = dynamical_system.transition(space="task", x_t=x_t)["desired state"]
        if order == "1st":
            visited_norm[t] = x_t.detach().cpu().numpy()
        else:
            visited_norm[t] = x_t[:, :dim].detach().cpu().numpy()

    return _denormalize_positions(visited_norm, data)


def plot_learned_ds_static(
    visited_pos: np.ndarray,
    demonstration_xy: np.ndarray,
    goal: np.ndarray,
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
    else:
        fig = ax.figure

    ax.plot(
        demonstration_xy[:, 0],
        demonstration_xy[:, 1],
        color="lightgray",
        linewidth=6,
        label="Demonstration",
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
    ax.plot(goal[0], goal[1], "g*", markersize=15, label="Goal", zorder=12)
    xlim, ylim = _axis_limits(demonstration_xy, visited_pos)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    return fig


def _overlay_demonstration_on_axes(ax: plt.Axes, demonstration_xy: np.ndarray) -> None:
    ax.plot(
        demonstration_xy[:, 0],
        demonstration_xy[:, 1],
        color="lightgray",
        linewidth=6,
        label="Demonstration",
        zorder=5,
    )


def run_learned_ds_pipeline(
    order: OrderTag,
    *,
    train: bool = True,
    simulate: bool = True,
    force_train: bool = False,
    max_iterations: int | None = None,
    n_simulate: int | None = None,
    n_steps: int = 2000,
    live_plot: bool = True,
    pause_time: float = 1e-5,
    save_path: str | None = None,
) -> dict:
    """Train (optional), simulate, and plot learned DS."""
    if train:
        learner, _, data, _ = train_learned_ds(
            order,
            max_iterations=max_iterations,
            force=force_train,
            n_simulate=n_simulate,
            n_steps=n_steps,
        )
    else:
        learner, _, data, _ = load_learned_ds(order)

    goal = np.asarray(data["goals"][0], dtype=float).reshape(-1)
    demo_xy = demonstration_xy_from_data(data)
    visited_pos = None
    if simulate:
        visited_pos = simulate_learned_ds(learner, data, order, n_steps=n_steps)

    title = PLOT_TITLE_BY_ORDER[order]
    fig = None
    if simulate and live_plot:
        fig = plt.figure(figsize=(8, 8))
        fig.show()
        plotter = TrajectoryPlotter(
            fig,
            x0=visited_pos[0].T,
            pause_time=pause_time,
            goal=goal,
        )
        _overlay_demonstration_on_axes(plotter._ax, demo_xy)
        plotter._ax.set_title(title)
        xlim, ylim = _axis_limits(demo_xy, visited_pos)
        plotter._ax.set_xlim(xlim)
        plotter._ax.set_ylim(ylim)
        for t in range(1, visited_pos.shape[0]):
            plotter.update(visited_pos[t].T)
            plotter._ax.set_xlim(xlim)
            plotter._ax.set_ylim(ylim)
        plotter.draw()
    if simulate and save_path:
        fig = plot_learned_ds_static(visited_pos, demo_xy, goal, title)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if simulate and live_plot:
        plt.show()

    if simulate and visited_pos is not None:
        final = visited_pos[-1]
        pos_err = np.max(np.linalg.norm(final - goal, axis=1))
        print(f"Goal: {goal}, max final position error: {pos_err:.4e}")

    return {
        "learner": learner,
        "data": data,
        "visited_pos": visited_pos,
        "goal": goal,
        "demonstration": demo_xy,
    }
