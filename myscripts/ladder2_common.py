"""
Shared utilities for ladder step 2: learned DS from one straight-line demonstration.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pumafabrics.puma_adapted.agent.neural_network import DEVICE
from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import denormalize_state
from pumafabrics.puma_adapted.initializer import initialize_framework
from pumafabrics.puma_adapted.tools.animation import TrajectoryPlotter

_MYSCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _MYSCRIPTS not in sys.path:
    sys.path.insert(0, _MYSCRIPTS)

from ladder1a import default_initial_states as default_initial_positions
from ladder1b import default_initial_states as default_initial_states_2nd_order
from ladder2_data import ensure_straight_line_dataset, load_demonstration_xy

OrderTag = Literal["1st", "2nd"]

PARAMS_BY_ORDER: dict[OrderTag, str] = {
    "1st": "ladder2_1st_order_2D",
    "2nd": "ladder2_2nd_order_2D",
}


def repo_results_base() -> str:
    return _REPO_ROOT + os.sep


def model_checkpoint_path(params) -> str:
    """Checkpoint path (params.results_path already ends with `<primitive_id>/`)."""
    return os.path.join(params.results_path, "model")


def build_params(order: OrderTag, load_model: bool = False, train: bool = True):
    module_name = f"pumafabrics.puma_adapted.params.{PARAMS_BY_ORDER[order]}"
    Params = getattr(importlib.import_module(module_name), "Params")
    params = Params(repo_results_base())
    params.results_path += params.selected_primitives_ids + "/"
    params.load_model = load_model
    params.train = train
    return params, PARAMS_BY_ORDER[order]


def train_learned_ds(
    order: OrderTag,
    max_iterations: int | None = None,
    force: bool = False,
    verbose: bool = True,
) -> tuple[object, object, dict, str]:
    """
    Train (or load) a PUMA model on the single straight-line demonstration.

    Returns learner, evaluator, data, params_module_name.
    """
    ensure_straight_line_dataset()
    params, params_name = build_params(order, load_model=False, train=True)
    if max_iterations is not None:
        params.max_iterations = max_iterations

    ckpt = model_checkpoint_path(params)
    if os.path.isfile(ckpt) and not force:
        if verbose:
            print(f"Checkpoint exists at {ckpt}; use --force-train to retrain.")
        return load_learned_ds(order)

    learner, evaluator, data = initialize_framework(params, params_name, verbose=verbose)

    log_name = f"{params_name}_{params.selected_primitives_ids}"
    writer = SummaryWriter(log_dir=os.path.join(repo_results_base(), "results", "tensorboard_runs", log_name))

    if verbose:
        print(f"Training {params_name} for up to {params.max_iterations} iterations...")
        print(f"Results path: {params.results_path}")
    t0 = time.perf_counter()
    for iteration in range(params.max_iterations + 1):
        if iteration % params.evaluation_interval == 0:
            metrics_acc, metrics_stab = evaluator.run(iteration=iteration)

            if params.save_evaluation:
                evaluator.save_progress(params.results_path, iteration, learner.model, writer)

            if verbose:
                print(
                    f"  iter {iteration}: metric sum={metrics_acc['metrics sum']:.4f}, "
                    f"spurious={metrics_stab['n spurious']}",
                )

        loss, loss_list, losses_names = learner.train_step()

        if verbose and iteration % 10 == 0:
            print(f"  iter {iteration}: total cost={loss.item():.6f}")

        for j in range(len(losses_names)):
            writer.add_scalar("losses/" + losses_names[j], loss_list[j], iteration)

    if verbose:
        print(f"Training finished in {time.perf_counter() - t0:.1f}s")

    writer.close()

    # Persist final weights (train.py only saves via save_progress on best eval;
    # this guarantees a loadable checkpoint after a full run).
    os.makedirs(params.results_path, exist_ok=True)
    torch.save(learner.model.state_dict(), ckpt)
    if verbose:
        print(f"Saved model to {ckpt}")

    return learner, evaluator, data, params_name


def load_learned_ds(order: OrderTag, verbose: bool = False):
    ensure_straight_line_dataset()
    params, params_name = build_params(order, load_model=True, train=False)
    ckpt = model_checkpoint_path(params)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"No trained model at {ckpt}. Run with --train or omit --simulate-only.",
        )
    learner, evaluator, data = initialize_framework(params, params_name, verbose=verbose)
    return learner, evaluator, data, params_name


def initial_states_for_order(order: OrderTag) -> np.ndarray:
    if order == "1st":
        return default_initial_positions()
    return default_initial_states_2nd_order()


def _denormalize_positions(pos_norm: np.ndarray, data: dict) -> np.ndarray:
    x_min = np.asarray(data["x min"]).reshape(-1)
    x_max = np.asarray(data["x max"]).reshape(-1)
    flat = pos_norm.reshape(-1, pos_norm.shape[-1])
    flat = denormalize_state(flat.T, x_min, x_max).T
    return flat.reshape(pos_norm.shape)


def simulate_learned_ds(
    learner,
    data: dict,
    order: OrderTag,
    n_steps: int = 2000,
    delta_t: float = 0.1,
) -> np.ndarray:
    """Roll out learned DS; returns denormalized position history (T+1, n_traj, 2)."""
    state_init = initial_states_for_order(order)
    x_min = data["x min"]
    x_max = data["x max"]
    # Grid initials are in task space; DS integrates in normalized coordinates.
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
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    return fig


def run_learned_ds_pipeline(
    order: OrderTag,
    *,
    train: bool = True,
    simulate: bool = True,
    force_train: bool = False,
    max_iterations: int | None = None,
    n_steps: int = 2000,
    live_plot: bool = True,
    pause_time: float = 1e-5,
    save_path: str | None = None,
) -> dict:
    """Train (optional), simulate, and plot learned DS."""
    if train:
        learner, _, data, _ = train_learned_ds(
            order, max_iterations=max_iterations, force=force_train,
        )
    else:
        learner, _, data, _ = load_learned_ds(order)

    goal = np.asarray(data["goals"][0], dtype=float).reshape(-1)
    demo_xy = load_demonstration_xy()
    visited_pos = None
    if simulate:
        visited_pos = simulate_learned_ds(learner, data, order, n_steps=n_steps)

    title = f"Learned {'1st' if order == '1st' else '2nd'}-order DS (straight-line demo)"
    fig = None
    if simulate and live_plot:
        pos_init = initial_states_for_order(order)
        if order == "2nd":
            pos_init = pos_init[:, :2]
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.show()
        plotter = TrajectoryPlotter(fig, x0=pos_init.T, pause_time=pause_time, goal=goal)
        for t in range(visited_pos.shape[0]):
            plotter.update(visited_pos[t].T)
    elif simulate and save_path:
        fig = plot_learned_ds_static(visited_pos, demo_xy, goal, title)

    if simulate and save_path:
        if fig is None:
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
