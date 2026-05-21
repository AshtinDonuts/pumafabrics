"""LAIR capricorn episode 0 export for ladder step 4 (one self-intersecting trajectory)."""

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
        "ladder4_capricorn",
    )
)
_PRIMITIVE_DIR = os.path.join(_DATASET_ROOT, "capricorn")
_DEMO_PATH = os.path.join(_PRIMITIVE_DIR, "demo_0.npy")

_SOURCE_NPY = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "pumafabrics",
        "puma_adapted",
        "datasets",
        "LAIR",
        "capricorn",
        "capricorn_0.npy",
    )
)

DEFAULT_EPISODE_INDEX = 0


def load_capricorn_demo_xy(
    npy_path: str = _SOURCE_NPY,
    episode_index: int = DEFAULT_EPISODE_INDEX,
) -> np.ndarray:
    """Load one LAIR capricorn demonstration as (T, 2) positions."""
    if episode_index != 0:
        base = os.path.join(os.path.dirname(npy_path), "capricorn")
        npy_path = os.path.join(base, f"capricorn_{episode_index}.npy")
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Capricorn source not found: {npy_path}")

    data = np.load(npy_path)
    if data.shape[0] == 1:
        data = data[0]
    return np.asarray(data, dtype=float)


def ensure_capricorn_dataset(
    source_npy: str = _SOURCE_NPY,
    episode_index: int = DEFAULT_EPISODE_INDEX,
    overwrite: bool = False,
) -> str:
    """
    Write `datasets/ladder4_capricorn/capricorn/demo_0.npy` from LAIR episode 0.

    Returns the dataset root directory passed to PUMA as `dataset_name`.
    """
    if os.path.isfile(_DEMO_PATH) and not overwrite:
        return "ladder4_capricorn"

    trajectory = load_capricorn_demo_xy(source_npy, episode_index)
    # LAIR convention: (1, D, T) before loader transpose -> store (1, T, D)
    array = trajectory[np.newaxis, :, :]
    os.makedirs(_PRIMITIVE_DIR, exist_ok=True)
    np.save(_DEMO_PATH, array)
    return "ladder4_capricorn"


def demo_path() -> str:
    return _DEMO_PATH


def load_demonstration_xy() -> np.ndarray:
    """Load the capricorn demo as (T, 2)."""
    ensure_capricorn_dataset()
    data = np.load(_DEMO_PATH)
    if data.shape[0] == 1:
        data = data[0]
    return np.asarray(data, dtype=float)


if __name__ == "__main__":
    path = ensure_capricorn_dataset(overwrite=True)
    pts = load_demonstration_xy()
    print(
        f"Wrote dataset '{path}' ({pts.shape[0]} points), "
        f"start={pts[0]}, goal={pts[-1]}",
    )
