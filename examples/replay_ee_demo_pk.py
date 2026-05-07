import argparse
import os
import pickle
import time
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    raise SystemExit(f"Missing dependency pybullet: {e}")


def _rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert 3x3 rotation matrix to quaternion (x, y, z, w).
    Assumes R is a proper rotation matrix.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"Expected (3,3) rotation matrix, got {R.shape}")

    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _load_demo(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict pickle, got {type(obj)}")

    if "x_pos" not in obj:
        raise KeyError(f"Missing key 'x_pos' in {path}. Keys: {sorted(obj.keys())}")

    x_pos = np.asarray(obj["x_pos"], dtype=float)
    if x_pos.ndim != 2 or x_pos.shape[1] != 3:
        raise ValueError(f"Expected x_pos shape (T,3), got {x_pos.shape}")

    x_rot = None
    if "x_rot" in obj and obj["x_rot"] is not None:
        xr = np.asarray(obj["x_rot"], dtype=float)
        if xr.shape == (x_pos.shape[0], 3, 3):
            x_rot = xr

    dt = 0.01
    if "delta_t" in obj and obj["delta_t"] is not None:
        d = np.asarray(obj["delta_t"], dtype=float)
        if d.size:
            dt = float(np.median(d))
    return x_pos, x_rot, dt


def _iter_waypoints(
    x_pos: np.ndarray, x_rot: Optional[np.ndarray]
) -> Iterable[Tuple[np.ndarray, Optional[Tuple[float, float, float, float]]]]:
    if x_rot is None:
        for pos in x_pos:
            yield pos, None
    else:
        for pos, R in zip(x_pos, x_rot):
            yield pos, _rotmat_to_quat_xyzw(R)


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay ee_state_*.pk in raw PyBullet using IK.")
    ap.add_argument("pk_path", help="Path to ee_state_*.pk (expects keys x_pos, optionally x_rot).")
    ap.add_argument("--urdf", default=None, help="URDF path. Defaults to pumafabrics KUKA iiwa14 URDF.")
    ap.add_argument("--end_effector_link", default="iiwa_link_7", help="End-effector link name (for readability only).")
    ap.add_argument("--end_effector_link_index", type=int, default=6, help="PyBullet link index used for IK.")
    ap.add_argument("--gui", action="store_true", help="Use PyBullet GUI.")
    ap.add_argument("--sleep", action="store_true", help="Sleep real-time according to demo dt.")
    ap.add_argument("--pos_gain", type=float, default=0.2, help="Position control gain.")
    ap.add_argument("--max_force", type=float, default=200.0, help="Max joint motor force.")
    ap.add_argument("--subsample", type=int, default=1, help="Use every Nth waypoint.")
    args = ap.parse_args()

    x_pos, x_rot, dt = _load_demo(args.pk_path)

    if args.urdf is None:
        args.urdf = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../pumafabrics/tamed_puma/config/urdfs/iiwa14.urdf")
        )
    if not os.path.isfile(args.urdf):
        raise SystemExit(f"URDF not found: {args.urdf}")

    cid = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    flags = p.URDF_USE_INERTIA_FROM_FILE
    robot = p.loadURDF(args.urdf, useFixedBase=True, flags=flags)

    # Best-effort: control first 7 revolute joints (iiwa has 7 dof)
    joint_indices = []
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        jtype = info[2]
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            joint_indices.append(j)
    joint_indices = joint_indices[:7]

    # Simple visualization of the path
    if args.gui:
        for i in range(0, len(x_pos), max(1, len(x_pos) // 200)):
            p.addUserDebugPoints([x_pos[i].tolist()], [[0, 0, 1]], pointSize=2, lifeTime=0)

    target_quat_default = (0.0, 0.0, 0.0, 1.0)
    for i, (pos, quat) in enumerate(_iter_waypoints(x_pos[:: args.subsample], None if x_rot is None else x_rot[:: args.subsample])):
        quat = target_quat_default if quat is None else quat

        q_sol = p.calculateInverseKinematics(
            robot,
            args.end_effector_link_index,
            targetPosition=pos.tolist(),
            targetOrientation=quat,
            maxNumIterations=100,
            residualThreshold=1e-4,
        )
        q_cmd = [q_sol[j] for j in range(len(joint_indices))]

        p.setJointMotorControlArray(
            robot,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_cmd,
            positionGains=[args.pos_gain] * len(joint_indices),
            forces=[args.max_force] * len(joint_indices),
        )

        # Step a few internal steps to let motors move (tuned for visual replay, not control fidelity)
        for _ in range(4):
            p.stepSimulation()
            if args.sleep and args.gui:
                time.sleep(dt / 4.0)

        if args.gui and (i % 10 == 0):
            p.addUserDebugPoints([pos.tolist()], [[1, 0, 0]], pointSize=4, lifeTime=0.2)

    p.disconnect(cid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

