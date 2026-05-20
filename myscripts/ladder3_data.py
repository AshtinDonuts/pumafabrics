"""LASA heee.mat demo export for ladder step 3 (one curved trajectory)."""

from __future__ import annotations

import os

import numpy as np

try:
    import scipy.io as sio
except ImportError as e:
    raise ImportError(
        "scipy is required to read LASA .mat files (e.g. `pip install scipy`).",
    ) from e

# Repo-relative dataset root (LAIR-style layout for load_numpy_file).
_DATASET_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "pumafabrics",
        "puma_adapted",
        "datasets",
        "ladder3_heee",
    )
)
_PRIMITIVE_DIR = os.path.join(_DATASET_ROOT, "heee")
_DEMO_PATH = os.path.join(_PRIMITIVE_DIR, "demo_0.npy")

_SOURCE_MAT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "pumafabrics",
        "puma_adapted",
        "datasets",
        "LASA",
        "heee.mat",
    )
)

DEFAULT_DEMO_INDEX = 0


def load_lasa_demo_xy(mat_path: str, demo_index: int = DEFAULT_DEMO_INDEX) -> np.ndarray:
    """Load one LASA demonstration as (T, 2) positions."""
    mat_file = sio.loadmat(mat_path)
    if "demos" not in mat_file:
        raise KeyError(f"No 'demos' variable in {mat_path}")
    data = mat_file["demos"]
    if demo_index < 0 or demo_index >= data.shape[1]:
        raise IndexError(
            f"demo_index {demo_index} out of range for {data.shape[1]} demos in {mat_path}",
        )
    sx = np.asarray(data[0, demo_index]["pos"][0, 0][0], dtype=float).reshape(-1)
    sy = np.asarray(data[0, demo_index]["pos"][0, 0][1], dtype=float).reshape(-1)
    n = min(sx.size, sy.size)
    return np.column_stack([sx[:n], sy[:n]])


def ensure_heee_dataset(
    mat_path: str = _SOURCE_MAT,
    demo_index: int = DEFAULT_DEMO_INDEX,
    overwrite: bool = False,
) -> str:
    """
    Write `datasets/ladder3_heee/heee/demo_0.npy` from the chosen LASA demo.

    Returns the dataset root directory passed to PUMA as `dataset_name`.
    """
    if os.path.isfile(_DEMO_PATH) and not overwrite:
        return "ladder3_heee"

    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"LASA source not found: {mat_path}")

    trajectory = load_lasa_demo_xy(mat_path, demo_index)
    # LAIR convention: (1, D, T) before loader transpose -> store (1, T, D)
    array = trajectory[np.newaxis, :, :]
    os.makedirs(_PRIMITIVE_DIR, exist_ok=True)
    np.save(_DEMO_PATH, array)
    return "ladder3_heee"


def demo_path() -> str:
    return _DEMO_PATH


def load_demonstration_xy() -> np.ndarray:
    """Load the curved heee demo as (T, 2)."""
    ensure_heee_dataset()
    data = np.load(_DEMO_PATH)
    if data.shape[0] == 1:
        data = data[0]
    return np.asarray(data, dtype=float)


if __name__ == "__main__":
    path = ensure_heee_dataset(overwrite=True)
    pts = load_demonstration_xy()
    print(
        f"Wrote dataset '{path}' ({pts.shape[0]} points), "
        f"start={pts[0]}, goal={pts[-1]}",
    )
