"""Synthetic straight-line demonstration for ladder step 2."""

from __future__ import annotations

import os

import numpy as np

# Repo-relative dataset root (LAIR-style layout for load_numpy_file).
_DATASET_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "pumafabrics",
        "puma_adapted",
        "datasets",
        "ladder2_line",
    )
)
_PRIMITIVE_DIR = os.path.join(_DATASET_ROOT, "straight_line")
_DEMO_PATH = os.path.join(_PRIMITIVE_DIR, "line_0.npy")

# Default segment (one demonstration, attractor at the end point).
DEFAULT_START = np.array([-0.9, -0.7], dtype=float)
DEFAULT_GOAL = np.array([0.9, 0.7], dtype=float)


def make_straight_line_trajectory(
    start: np.ndarray = DEFAULT_START,
    goal: np.ndarray = DEFAULT_GOAL,
    n_points: int = 500,
) -> np.ndarray:
    """Return (T, 2) positions along a straight segment."""
    start = np.asarray(start, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    t = np.linspace(0.0, 1.0, n_points)
    return start + t[:, None] * (goal - start)


def ensure_straight_line_dataset(
    start: np.ndarray = DEFAULT_START,
    goal: np.ndarray = DEFAULT_GOAL,
    n_points: int = 500,
    overwrite: bool = False,
) -> str:
    """
    Write `datasets/ladder2_line/straight_line/line_0.npy` if missing.

    Returns the dataset root directory passed to PUMA as `dataset_name`.
    """
    if os.path.isfile(_DEMO_PATH) and not overwrite:
        return "ladder2_line"

    trajectory = make_straight_line_trajectory(start, goal, n_points)
    # LAIR convention: (1, D, T) before loader transpose -> store (1, T, D)
    array = trajectory[np.newaxis, :, :]
    os.makedirs(_PRIMITIVE_DIR, exist_ok=True)
    np.save(_DEMO_PATH, array)
    return "ladder2_line"


def demo_path() -> str:
    return _DEMO_PATH


def load_demonstration_xy() -> np.ndarray:
    """Load the straight-line demo as (T, 2)."""
    ensure_straight_line_dataset()
    data = np.load(_DEMO_PATH)
    if data.shape[0] == 1:
        data = data[0]
    return np.asarray(data, dtype=float)


if __name__ == "__main__":
    path = ensure_straight_line_dataset(overwrite=True)
    pts = load_demonstration_xy()
    print(f"Wrote dataset '{path}' ({pts.shape[0]} points), goal={pts[-1]}")
