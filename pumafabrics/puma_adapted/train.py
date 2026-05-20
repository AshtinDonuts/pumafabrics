from __future__ import annotations

import importlib
import os
import time
from typing import Any, Callable, List, Optional

TrainingStepCallback = Callable[[int, Any, Any, dict], None]

from simple_parsing import ArgumentParser  # pyright: ignore[reportMissingImports]
from pumafabrics.puma_adapted.initializer import initialize_framework
from torch.utils.tensorboard import SummaryWriter


def import_params_class(params_module: str) -> type:
    """Load a Params dataclass from a params module name (no package prefix)."""
    last_error: Optional[ModuleNotFoundError] = None
    for prefix in ("pumafabrics.puma_adapted.params.", "params."):
        try:
            return getattr(importlib.import_module(prefix + params_module), "Params")
        except ModuleNotFoundError as exc:
            last_error = exc
    raise ModuleNotFoundError(
        f"Could not import params module {params_module!r}.",
    ) from last_error


def configure_params(
    params_module: str,
    results_base_directory: str,
    *,
    selected_primitives_ids: Optional[str] = None,
    results_path: Optional[str] = None,
    boundary_loss_weight: Optional[float] = None,
    max_iterations: Optional[int] = None,
) -> tuple[Any, str]:
    """
    Instantiate Params and apply CLI-style overrides.

    Returns (params, params_module) with results_path ending in ``<primitive_id>/``.
    """
    Params = import_params_class(params_module)
    params = Params(results_base_directory)
    if selected_primitives_ids is not None:
        params.selected_primitives_ids = selected_primitives_ids.strip()
    if boundary_loss_weight is not None:
        params.boundary_loss_weight = boundary_loss_weight
    if max_iterations is not None:
        params.max_iterations = max_iterations
    if results_path is not None:
        rel = results_path.strip()
        if rel and not rel.endswith("/"):
            rel += "/"
        params.results_path = results_base_directory + rel
    params.results_path += params.selected_primitives_ids + "/"
    return params, params_module


def run_training(
    params: Any,
    params_module_name: str,
    *,
    tensorboard_log_dir: Optional[str] = None,
    start_iteration: int = 0,
    step_callback: Optional[TrainingStepCallback] = None,
    step_callback_interval: Optional[int] = None,
    verbose: bool = True,
) -> tuple[Any, Any, dict, float]:
    """
    Run the standard PUMA training loop.

    Parameters
    ----------
    start_iteration : int
        First iteration index to run (use > 0 with ``params.load_model=True`` to resume).
    step_callback : callable, optional
        ``callback(iteration, learner, evaluator, data)`` invoked during training.
    step_callback_interval : int, optional
        Call ``step_callback`` when ``iteration % step_callback_interval == 0``.

    Returns (learner, evaluator, data, elapsed_seconds).
    """
    learner, evaluator, data = initialize_framework(params, params_module_name, verbose=verbose)

    log_name = f"{params_module_name}_{params.selected_primitives_ids}"
    if tensorboard_log_dir is None:
        tensorboard_log_dir = os.path.join("results", "tensorboard_runs", log_name)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    if verbose:
        if start_iteration > 0:
            print(
                f"Resuming {params_module_name} from iteration {start_iteration} "
                f"to {params.max_iterations}...",
            )
        else:
            print(f"Training {params_module_name} for up to {params.max_iterations} iterations...")
        print(f"Results path: {params.results_path}")

    t0 = time.perf_counter()
    for iteration in range(start_iteration, params.max_iterations + 1):
        if iteration % params.evaluation_interval == 0:
            metrics_acc, metrics_stab = evaluator.run(iteration=iteration)

            if params.save_evaluation:
                evaluator.save_progress(params.results_path, iteration, learner.model, writer)

            if verbose:
                print(
                    "Metrics sum:",
                    metrics_acc["metrics sum"],
                    "; Number of unsuccessful trajectories:",
                    metrics_stab["n spurious"],
                )

        loss, loss_list, losses_names = learner.train_step()

        if verbose and iteration % 10 == 0:
            print(iteration, "Total cost:", loss.item())

        for j in range(len(losses_names)):
            writer.add_scalar("losses/" + losses_names[j], loss_list[j], iteration)

        if (
            step_callback is not None
            and step_callback_interval is not None
            and step_callback_interval > 0
            and iteration % step_callback_interval == 0
        ):
            step_callback(iteration, learner, evaluator, data)

    elapsed = time.perf_counter() - t0
    writer.close()

    if verbose:
        print(f"Training finished in {elapsed:.1f}s")

    return learner, evaluator, data, elapsed


def main(argv: Optional[List[str]] = None) -> None:
    parser = ArgumentParser()
    parser.add_argument("--params", type=str, default="2nd_order_R3S3_kinova", help="")
    parser.add_argument("--results-base-directory", type=str, default="./", help="")
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="If set, replaces the params module results_path (relative segment only); "
        "final path is results_base_directory + this + selected_primitives_ids/.",
    )
    parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=None,
        help="If set, overrides params.boundary_loss_weight after loading the params module.",
    )
    parser.add_argument(
        "--selected-primitives-ids",
        type=str,
        default=None,
        help='If set, overrides params.selected_primitives_ids (comma-separated indices from '
        'dataset_keys.py, e.g. "0" or "4,0,6").',
    )
    args = parser.parse_args(argv)

    params, params_module_name = configure_params(
        args.params,
        args.results_base_directory,
        selected_primitives_ids=args.selected_primitives_ids,
        results_path=args.results_path,
        boundary_loss_weight=args.boundary_loss_weight,
    )

    time1 = time.perf_counter()
    run_training(params, params_module_name)
    time2 = time.perf_counter()
    print("timer:", time2 - time1)


if __name__ == "__main__":
    main()
