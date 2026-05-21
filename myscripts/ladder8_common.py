"""
Shared utilities for ladder step 8: keypoint policy transport.

Follows the policy_transportation API (residual form, default):

    phi(x) = gamma(x) + psi(gamma(x)),    J_phi = J_gamma + J_psi @ J_gamma
    v_hat = J_phi(x) @ v

with gamma an affine map fit from source/target keypoints and psi a learned
nonlinear delta field. Only position and velocity are transported (no orientation).

Reference: policy_transportation.transportation.PolicyTransportation
https://github.com/franzesegiovanni/policy_transportation/blob/devel/policy_transportation/transportation/transportation.py
"""

from __future__ import annotations

import os
import sys
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
from ladder5_common import (
    SOURCE_CHOICES,
    SOURCE_LABELS,
    SourceTag,
    _demonstration_xy,
    _load_learned_source,
    learned_derivative_in_source_frame,
)

OrderTag = Literal["1st", "2nd"]


def _as_batch(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        return points.reshape(1, -1)
    return points


class AffineTransform2D:
    """Affine gamma(x) = M @ x + t fit from paired keypoints (2D Procrustes)."""

    def __init__(self) -> None:
        self.matrix: np.ndarray = np.eye(2)
        self.translation: np.ndarray = np.zeros(2)

    @classmethod
    def from_rotation_scale_translation(
        cls,
        angle_rad: float,
        scale: np.ndarray,
        translation: np.ndarray,
    ) -> AffineTransform2D:
        scale_arr = np.asarray(scale, dtype=float).reshape(-1)
        if scale_arr.size == 1:
            scale_arr = np.full(2, float(scale_arr[0]))
        c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
        rotation = np.array([[c, -s], [s, c]], dtype=float)
        obj = cls()
        obj.matrix = np.diag(scale_arr) @ rotation
        obj.translation = np.asarray(translation, dtype=float).reshape(2)
        return obj

    def fit(
        self,
        source: np.ndarray,
        target: np.ndarray,
        *,
        do_scale: bool = True,
        do_rotation: bool = True,
    ) -> None:
        source = _as_batch(source)
        target = _as_batch(target)
        if source.shape != target.shape:
            raise ValueError("source and target must have the same shape.")
        mu_s = source.mean(axis=0)
        mu_t = target.mean(axis=0)
        xs = source - mu_s
        yt = target - mu_t
        n = source.shape[0]

        if do_rotation:
            h = xs.T @ yt
            u, _, vt = np.linalg.svd(h)
            r = vt.T @ u.T
            if np.linalg.det(r) < 0:
                vt = vt.copy()
                vt[-1, :] *= -1.0
                r = vt.T @ u.T
        else:
            r = np.eye(2)

        if do_scale:
            var_s = (xs**2).sum() / max(n, 1)
            scale = float(np.trace(r @ xs.T @ yt) / max(var_s, 1e-12))
        else:
            scale = 1.0

        self.matrix = scale * r
        self.translation = mu_t - self.matrix @ mu_s

    def predict(self, pos: np.ndarray) -> np.ndarray:
        pos = _as_batch(pos)
        return (self.matrix @ pos.T).T + self.translation

    def derivative(self, pos: np.ndarray) -> np.ndarray:
        """J_gamma(x) = M, batched as (N, 2, 2)."""
        pos = _as_batch(pos)
        n = pos.shape[0]
        return np.tile(self.matrix, (n, 1, 1))


class ParametricDeltaRegressor2D:
    """
    Nonlinear residual psi(z) = delta(z) with bounded trig basis (fits amplitudes at keypoints).

    Matches policy_transportation residual psi in phi = gamma + psi(gamma), not psi = z + delta.
    """

    def __init__(self, omega: np.ndarray, *, seed: int = 0) -> None:
        self.omega = np.asarray(omega, dtype=float).reshape(2)
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
        self.amplitude = np.zeros(2)

    def _basis(self, z: np.ndarray) -> np.ndarray:
        z = _as_batch(z)
        theta0 = self.omega[0] * z[:, 0] + self.phase[0]
        theta1 = self.omega[1] * z[:, 1] + self.phase[1]
        return np.column_stack(
            [
                np.sin(theta0) * np.cos(theta1),
                np.cos(theta0) * np.sin(theta1),
            ]
        )

    def _basis_jacobian(self, z: np.ndarray) -> np.ndarray:
        """d(delta)/dz for delta = diag(amplitude) @ g(z), shape (N, 2, 2)."""
        z = _as_batch(z)
        n = z.shape[0]
        if np.all(self.amplitude == 0.0):
            return np.zeros((n, 2, 2))
        theta0 = self.omega[0] * z[:, 0] + self.phase[0]
        theta1 = self.omega[1] * z[:, 1] + self.phase[1]
        w0, w1 = self.omega[0], self.omega[1]
        a0, a1 = self.amplitude[0], self.amplitude[1]
        c0, s0 = np.cos(theta0), np.sin(theta0)
        c1, s1 = np.cos(theta1), np.sin(theta1)
        j00 = a0 * w0 * c0 * c1
        j01 = -a0 * w1 * s0 * s1
        j10 = -a1 * w0 * s0 * s1
        j11 = a1 * w1 * c0 * c1
        return np.stack(
            [np.stack([j00, j01], axis=1), np.stack([j10, j11], axis=1)],
            axis=1,
        )

    def fit(self, inputs: np.ndarray, deltas: np.ndarray) -> None:
        inputs = _as_batch(inputs)
        deltas = _as_batch(deltas)
        g = self._basis(inputs)
        self.amplitude = np.zeros(2)
        for d in range(2):
            gd = g[:, d]
            denom = np.dot(gd, gd)
            if denom > 1e-12:
                self.amplitude[d] = float(np.dot(gd, deltas[:, d]) / denom)

    def predict(self, pos: np.ndarray, *, return_std: bool = False) -> np.ndarray:
        pos = _as_batch(pos)
        mean = self._basis(pos) * self.amplitude
        if return_std:
            return mean, np.zeros_like(mean)
        return mean

    def derivative(self, pos: np.ndarray, *, return_var: bool = False) -> np.ndarray:
        pos = _as_batch(pos)
        jac = self._basis_jacobian(pos)
        if return_var:
            return jac, np.zeros_like(jac)
        return jac


class PolicyTransportation2D:
    """
    Residual policy transportation map (2D), mirroring PolicyTransportation.

    phi(x) = gamma(x) + psi(gamma(x)),  J_phi = J_gamma + J_psi @ J_gamma
    """

    def __init__(self, *, is_residual: bool = True) -> None:
        self.is_residual = is_residual
        self.affine_transform = AffineTransform2D()
        self.nonlinear_transform: ParametricDeltaRegressor2D | None = None
        self.accuracy: float | None = None

    def fit(
        self,
        source_distribution: np.ndarray,
        target_distribution: np.ndarray,
        *,
        do_scale: bool = True,
        do_rotation: bool = True,
    ) -> None:
        source_distribution = _as_batch(source_distribution)
        target_distribution = _as_batch(target_distribution)
        if source_distribution.shape[0] != target_distribution.shape[0]:
            raise ValueError("source and target distributions must have the same number of points.")
        self.affine_transform.fit(
            source_distribution,
            target_distribution,
            do_scale=do_scale,
            do_rotation=do_rotation,
        )
        source_rotated = self.affine_transform.predict(source_distribution)
        if self.nonlinear_transform is not None:
            if self.is_residual:
                delta = target_distribution - source_rotated
                self.nonlinear_transform.fit(source_rotated, delta)
            else:
                self.nonlinear_transform.fit(source_rotated, target_distribution)

        transported = self.transport(source_distribution, return_std=False)
        self.accuracy = float(
            np.sqrt(np.mean(np.sum((transported - target_distribution) ** 2, axis=1)))
        )

    def transport(self, pos: np.ndarray, *, return_std: bool = False) -> np.ndarray:
        pos = _as_batch(pos)
        pos_rotated = self.affine_transform.predict(pos)
        if self.nonlinear_transform is None:
            if return_std:
                return pos_rotated, np.zeros_like(pos_rotated)
            return pos_rotated

        if return_std:
            delta_mean, delta_std = self.nonlinear_transform.predict(pos_rotated, return_std=True)
        else:
            delta_mean = self.nonlinear_transform.predict(pos_rotated, return_std=False)

        if self.is_residual:
            pos_transported = pos_rotated + delta_mean
        else:
            pos_transported = delta_mean

        if return_std:
            return pos_transported, delta_std
        return pos_transported

    def compute_jacobian(self, pos: np.ndarray, *, return_var: bool = False) -> np.ndarray:
        pos = _as_batch(pos)
        pos_rotated = self.affine_transform.predict(pos)
        j_gamma = self.affine_transform.derivative(pos)
        if self.nonlinear_transform is None:
            if return_var:
                return j_gamma, np.zeros_like(j_gamma)
            return j_gamma

        if return_var:
            j_psi, j_psi_var = self.nonlinear_transform.derivative(pos_rotated, return_var=True)
        else:
            j_psi = self.nonlinear_transform.derivative(pos_rotated, return_var=False)

        if self.is_residual:
            j_phi = j_gamma + j_psi @ j_gamma
        else:
            j_phi = j_psi @ j_gamma

        if return_var:
            return j_phi, j_psi_var
        return j_phi

    def transport_velocity(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        *,
        return_var: bool = False,
    ) -> np.ndarray:
        """v_hat = J_phi(pos) @ v (source-frame pos and vel)."""
        pos = _as_batch(pos)
        vel = _as_batch(vel)
        if return_var:
            j_phi, _ = self.compute_jacobian(pos, return_var=True)
        else:
            j_phi = self.compute_jacobian(pos, return_var=False)
        vel_t = (j_phi @ vel[..., np.newaxis])[..., 0]
        if return_var:
            return vel_t, np.zeros_like(vel_t)
        return vel_t

    def inverse_position(self, y: np.ndarray, *, max_iter: int = 20) -> np.ndarray:
        """Approximate x such that transport(x) = y."""
        y = _as_batch(y)
        x = y.copy()
        for _ in range(max_iter):
            x = x - (self.transport(x, return_std=False) - y)
        return x

    def pull_back_velocity(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Source velocity v from target position/velocity (v = J^{-1} v_hat)."""
        pos = _as_batch(pos)
        vel = _as_batch(vel)
        x = self.inverse_position(pos)
        j_phi = self.compute_jacobian(x)
        return np.linalg.solve(j_phi, vel)


def _resample_polyline(xy: np.ndarray, n_pts: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 2:
        return np.tile(xy[0], (n_pts, 1))
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-12:
        return np.tile(xy[0], (n_pts, 1))
    s = np.linspace(0.0, cum[-1], n_pts)
    out = np.zeros((n_pts, 2))
    for d in range(2):
        out[:, d] = np.interp(s, cum, xy[:, d])
    return out


def _synthetic_line_keypoints(n_pts: int = 40) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n_pts)
    return np.column_stack([t, 0.25 * t])


def _generative_policy_transport(
    *,
    angle_rad: float,
    scale: np.ndarray,
    translation: np.ndarray,
    deform_amplitude: float,
    deform_omega: np.ndarray,
    deform_seed: int,
) -> PolicyTransportation2D:
    """Ground-truth transport used to synthesize target keypoints before re-fitting."""
    gen = PolicyTransportation2D(is_residual=True)
    gen.affine_transform = AffineTransform2D.from_rotation_scale_translation(
        angle_rad,
        scale,
        translation,
    )
    gen.nonlinear_transform = ParametricDeltaRegressor2D(deform_omega, seed=deform_seed)
    gen.nonlinear_transform.amplitude = np.full(2, float(deform_amplitude))
    return gen


def generate_random_transport(
    *,
    deform_amplitude: float = 0.15,
    seed: int | None = None,
) -> PolicyTransportation2D:
    """Return a known (analytically defined) random transport map for testing.

    Composed of:
      1. A random affine transformation (random rotation, isotropic scale, translation).
      2. A random nonlinear residual delta field with a user-controlled amplitude.

    The map is *not* fitted from data; it is constructed directly and assumed known,
    matching the ``is_residual=True`` formulation of PolicyTransportation:

        phi(x) = gamma(x) + psi(gamma(x)),   J_phi = J_gamma + J_psi @ J_gamma

    Parameters
    ----------
    deform_amplitude:
        Amplitude of the nonlinear residual (controls the shape deformation).
        Set to 0 to get a pure affine map.
    seed:
        Integer seed for reproducible random generation.  ``None`` = non-reproducible.
    """
    rng = np.random.default_rng(seed)

    angle_rad = rng.uniform(-np.pi / 3.0, np.pi / 3.0)
    scale = rng.uniform(0.7, 1.5)
    translation = rng.uniform(-1.0, 1.0, size=2)
    omega = rng.uniform(0.5, 3.0, size=2)
    nl_seed = int(rng.integers(0, 2**31))

    transport = PolicyTransportation2D(is_residual=True)
    transport.affine_transform = AffineTransform2D.from_rotation_scale_translation(
        angle_rad, np.full(2, scale), translation
    )
    transport.nonlinear_transform = ParametricDeltaRegressor2D(omega, seed=nl_seed)
    transport.nonlinear_transform.amplitude = np.full(2, float(deform_amplitude))
    transport.accuracy = 0.0
    return transport


def source_keypoints_for_fit(
    source: SourceTag,
    data: dict | None,
    *,
    n_pts: int = 40,
) -> np.ndarray:
    demo = _demonstration_xy(source, data)
    if demo is not None:
        return _resample_polyline(demo, n_pts)
    return _synthetic_line_keypoints(n_pts)


def build_fitted_transport(
    source: SourceTag,
    data: dict | None,
    *,
    angle_rad: float,
    scale: np.ndarray,
    translation: np.ndarray,
    deform_amplitude: float,
    deform_omega: np.ndarray,
    deform_seed: int,
    n_keypoints: int = 40,
) -> PolicyTransportation2D:
    """Fit PolicyTransportation2D from source demo keypoints and a synthetic target warp."""
    source_pts = source_keypoints_for_fit(source, data, n_pts=n_keypoints)
    transport = _generative_policy_transport(
        angle_rad=angle_rad,
        scale=scale,
        translation=translation,
        deform_amplitude=deform_amplitude,
        deform_omega=deform_omega,
        deform_seed=deform_seed,
    )
    # Ladder warp is defined analytically (gamma from CLI + parametric psi). When paired
    # source/target keypoints come from demonstrations, call PolicyTransportation2D.fit().
    target_pts = transport.transport(source_pts)
    transport.accuracy = float(
        np.sqrt(np.mean(np.sum((target_pts - target_pts) ** 2, axis=1)))
    )
    return transport


def _simulate_learned_in_source_then_map(
    learner,
    data: dict,
    order: OrderTag,
    transport: PolicyTransportation2D,
    init_target: np.ndarray,
    n_steps: int,
    *,
    delta_t: float = 0.1,
) -> np.ndarray:
    """Roll out learned DS in source frame; map with transport / transport_velocity."""
    import torch
    from pumafabrics.puma_adapted.agent.neural_network import DEVICE
    from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import (
        denormalize_state,
        normalize_state,
    )

    init_target = np.asarray(init_target, dtype=float)
    if init_target.ndim == 1:
        init_target = init_target.reshape(1, -1)

    dim = 2
    n_traj = init_target.shape[0]
    x_min = data["x min"]
    x_max = data["x max"]

    if order == "1st":
        pos_src = transport.inverse_position(init_target)
        state_norm = normalize_state(pos_src, x_min, x_max)
        state_dim = dim
    else:
        q_tgt, v_tgt = init_target[:, :dim], init_target[:, dim:]
        q_src = transport.inverse_position(q_tgt)
        v_src = transport.pull_back_velocity(q_tgt, v_tgt)
        q_norm = normalize_state(q_src, x_min, x_max)
        state_norm = np.hstack([q_norm, v_src])
        state_dim = 2 * dim

    x_gpu = torch.FloatTensor(state_norm).to(DEVICE)
    dynamical_system = learner.init_dynamical_system(initial_states=x_gpu, delta_t=delta_t)

    visited = np.zeros((n_steps + 1, n_traj, state_dim))
    if order == "1st":
        visited[0] = transport.transport(pos_src)
    else:
        visited[0] = np.hstack(
            [
                transport.transport(q_src),
                transport.transport_velocity(q_src, v_src),
            ]
        )

    x_t = x_gpu
    for t in range(1, n_steps + 1):
        with torch.no_grad():
            x_t = dynamical_system.transition(space="task", x_t=x_t)["desired state"]
        x_np = x_t.detach().cpu().numpy()
        if order == "1st":
            pos_src = denormalize_state(
                np.asarray(x_np, dtype=float).reshape(-1, dim), x_min, x_max
            )
            visited[t] = transport.transport(pos_src)
        else:
            pos_src = denormalize_state(
                np.asarray(x_np[:, :dim], dtype=float).reshape(-1, dim), x_min, x_max
            )
            vel_src = denormalize_state(
                np.asarray(x_np[:, dim:], dtype=float).reshape(-1, dim),
                dynamical_system.min_vel.detach().cpu().numpy(),
                dynamical_system.max_vel.detach().cpu().numpy(),
            )
            visited[t] = np.hstack(
                [
                    transport.transport(pos_src),
                    transport.transport_velocity(pos_src, vel_src),
                ]
            )

    return visited


def transform_states_2nd(states: np.ndarray, transport: PolicyTransportation2D) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    q, v = states[:, :2], states[:, 2:]
    q_tgt = transport.transport(q)
    v_tgt = transport.transport_velocity(q, v)
    return np.hstack([q_tgt, v_tgt])


def default_initial_states_target(
    order: OrderTag,
    transport: PolicyTransportation2D,
    *,
    source: SourceTag = "analytic",
    data: dict | None = None,
) -> np.ndarray:
    if source in ("learned3", "learned4") and data is not None:
        from ladder3_common import initial_states_for_order as init_l3
        from ladder4_common import initial_states_for_order as init_l4

        init_fn = init_l3 if source == "learned3" else init_l4
        states_src = init_fn(order, data)
        if order == "1st":
            return transport.transport(states_src)
        return transform_states_2nd(states_src, transport)

    if order == "1st":
        return transport.transport(default_positions_1st())
    return transform_states_2nd(default_states_2nd(), transport)


class PolicyTransportedFirstOrderDS:
    """1st-order DS: integrate in source, dy/dt = J_phi(x) f(x) via transport_velocity."""

    def __init__(
        self,
        source: SourceTag,
        transport: PolicyTransportation2D,
        *,
        attractor: np.ndarray | None = None,
        gain: float = 1.0,
        learner=None,
        data: dict | None = None,
        dt: float = 0.01,
        delta_t: float = 0.1,
    ):
        self.source = source
        self.transport = transport
        self.dt = float(dt)
        self.delta_t = float(delta_t)
        self.dim = 2
        self.learner = learner
        self.data = data
        self._step_dt = dt if source == "analytic" else delta_t

        if source == "analytic":
            self._analytic = AnalyticStraightLineDS(
                attractor=np.asarray(attractor if attractor is not None else [0.0, 0.0]),
                gain=gain,
                dt=dt,
            )
        elif learner is None or data is None:
            raise ValueError("learned source requires learner and data.")

    @property
    def attractor_target(self) -> np.ndarray:
        if self.source == "analytic":
            src = self._analytic.attractor
        else:
            src = np.asarray(self.data["goals"][0], dtype=float).reshape(2)
        return self.transport.transport(src).reshape(2)

    def source_velocity(self, x: np.ndarray) -> np.ndarray:
        x = _as_batch(x)
        if self.source == "analytic":
            return self._analytic.velocity(x)
        return learned_derivative_in_source_frame(
            self.learner,
            self.data,
            "1st",
            x,
            delta_t=self.delta_t,
        )

    def simulate(self, y_init: np.ndarray, n_steps: int) -> np.ndarray:
        y_init = _as_batch(y_init)
        if self.source != "analytic":
            return _simulate_learned_in_source_then_map(
                self.learner,
                self.data,
                "1st",
                self.transport,
                y_init,
                n_steps,
                delta_t=self.delta_t,
            )
        n_traj = y_init.shape[0]
        x = self.transport.inverse_position(y_init)
        visited = np.zeros((n_steps + 1, n_traj, self.dim))
        visited[0] = self.transport.transport(x)
        for t in range(1, n_steps + 1):
            v_src = self.source_velocity(x)
            x = x + v_src * self._step_dt
            visited[t] = self.transport.transport(x)
        return visited


class PolicyTransportedSecondOrderDS:
    """2nd-order DS: position and velocity via transport / transport_velocity only."""

    def __init__(
        self,
        source: SourceTag,
        transport: PolicyTransportation2D,
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
        self.transport = transport
        self.dt = float(dt)
        self.delta_t = float(delta_t)
        self.dim = 2
        self.state_dim = 4
        self.learner = learner
        self.data = data
        self._step_dt = dt if source == "analytic" else delta_t

        if source == "analytic":
            self._analytic = AnalyticStraightLineDS2ndOrder(
                attractor=np.asarray(attractor if attractor is not None else [0.0, 0.0]),
                kp=kp,
                kd=kd,
                dt=dt,
            )
        elif learner is None or data is None:
            raise ValueError("learned source requires learner and data.")

    @property
    def attractor_target(self) -> np.ndarray:
        if self.source == "analytic":
            src = self._analytic.attractor
        else:
            src = np.asarray(self.data["goals"][0], dtype=float).reshape(2)
        return self.transport.transport(src).reshape(2)

    def split_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)
        return state[..., : self.dim], state[..., self.dim :]

    def stack_state(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.hstack([q, v])

    def source_acceleration(self, x: np.ndarray, v_src: np.ndarray) -> np.ndarray:
        x = _as_batch(x)
        v_src = _as_batch(v_src)
        if self.source == "analytic":
            return self._analytic.acceleration(x, v_src)
        _, a_src = learned_derivative_in_source_frame(
            self.learner,
            self.data,
            "2nd",
            self.stack_state(x, v_src),
            delta_t=self.delta_t,
        )
        return a_src

    def simulate(self, state_init: np.ndarray, n_steps: int) -> np.ndarray:
        state_init = _as_batch(state_init)
        if self.source != "analytic":
            return _simulate_learned_in_source_then_map(
                self.learner,
                self.data,
                "2nd",
                self.transport,
                state_init,
                n_steps,
                delta_t=self.delta_t,
            )
        n_traj = state_init.shape[0]
        q_tgt, v_tgt = self.split_state(state_init)
        q_src = self.transport.inverse_position(q_tgt)
        v_src = self.transport.pull_back_velocity(q_tgt, v_tgt)

        visited = np.zeros((n_steps + 1, n_traj, self.state_dim))
        visited[0] = self.stack_state(
            self.transport.transport(q_src),
            self.transport.transport_velocity(q_src, v_src),
        )

        q, v = q_src, v_src
        for t in range(1, n_steps + 1):
            a = self.source_acceleration(q, v)
            q = q + v * self._step_dt
            v = v + a * self._step_dt
            visited[t] = self.stack_state(
                self.transport.transport(q),
                self.transport.transport_velocity(q, v),
            )
        return visited

    def positions(self, visited: np.ndarray) -> np.ndarray:
        return visited[..., : self.dim]


def build_policy_transport_ds(
    order: OrderTag,
    source: SourceTag,
    transport: PolicyTransportation2D,
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
        return PolicyTransportedFirstOrderDS(
            source,
            transport,
            attractor=attractor,
            gain=gain,
            learner=learner,
            data=data,
            dt=dt,
            delta_t=delta_t,
        ), data
    return PolicyTransportedSecondOrderDS(
        source,
        transport,
        attractor=attractor,
        kp=kp,
        kd=kd,
        learner=learner,
        data=data,
        dt=dt,
        delta_t=delta_t,
    ), data


def plot_title(
    order: OrderTag,
    source: SourceTag,
    transport: PolicyTransportation2D,
    *,
    random_map: bool = False,
) -> str:
    order_label = "1st-order" if order == "1st" else "2nd-order"
    acc = transport.accuracy
    acc_str = f"{acc:.4g}" if acc is not None else "n/a"
    map_desc = "random gamma+psi (known)" if random_map else f"fitted gamma+psi (RMSE {acc_str})"
    return (
        f"Policy-transported {order_label} DS ({SOURCE_LABELS[source]})\n"
        f"{map_desc}"
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


def run_policy_transport_pipeline(
    order: OrderTag,
    *,
    source: SourceTag,
    angle_rad: float,
    scale: np.ndarray,
    translation: np.ndarray,
    deform_amplitude: float,
    deform_omega: np.ndarray,
    deform_seed: int,
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
    random_map: bool = False,
    map_seed: int | None = None,
) -> dict:
    learner, data = None, None
    if source != "analytic":
        _, _, data, _ = _load_learned_source(source, order)

    if random_map:
        transport = generate_random_transport(
            deform_amplitude=deform_amplitude,
            seed=map_seed,
        )
        print(
            f"[random map] seed={map_seed}, deform_amplitude={deform_amplitude}"
        )
        t_params = transport.affine_transform
        angle_deg = float(np.degrees(np.arctan2(t_params.matrix[1, 0], t_params.matrix[0, 0])))
        eff_scale = float(np.linalg.norm(t_params.matrix[:, 0]))
        print(
            f"  affine: rotation={angle_deg:.2f} deg, scale~{eff_scale:.3f}, "
            f"translation={t_params.translation}"
        )
    else:
        transport = build_fitted_transport(
            source,
            data,
            angle_rad=angle_rad,
            scale=scale,
            translation=translation,
            deform_amplitude=deform_amplitude,
            deform_omega=deform_omega,
            deform_seed=deform_seed,
        )

    ds, data = build_policy_transport_ds(
        order,
        source,
        transport,
        attractor=attractor,
        gain=gain,
        kp=kp,
        kd=kd,
        dt=dt,
        delta_t=delta_t,
    )

    state_init = default_initial_states_target(order, transport, source=source, data=data)
    if order == "1st":
        visited = ds.simulate(state_init, n_steps)
        visited_pos = visited
    else:
        visited = ds.simulate(state_init, n_steps)
        visited_pos = ds.positions(visited)

    goal_tgt = ds.attractor_target
    demo_src = _demonstration_xy(source, data)
    demo_tgt = transport.transport(demo_src) if demo_src is not None else None

    title = plot_title(order, source, transport, random_map=random_map)

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
    if random_map:
        print(f"Transport: randomly generated (seed={map_seed}, deform_amp={deform_amplitude}, accuracy={transport.accuracy:.4e})")
    else:
        print(
            f"Fitted transport (generative target): scale={scale}, translation={translation}, "
            f"rotation={np.degrees(angle_rad):.2f} deg, deform_amp={deform_amplitude}, "
            f"fit RMSE={transport.accuracy:.4e}"
        )
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
        "transport": transport,
    }


def add_policy_transport_cli_args(parser) -> None:
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
        default=[0.5, 0.3],
        metavar=("TX", "TY"),
        help="Generative affine translation for target keypoints (default: 0.5 0.3).",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=25.0,
        help="Generative affine rotation in degrees (default: 25).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[1.2],
        metavar=("S", "SY"),
        help="Generative affine scale (default: 1.2).",
    )
    parser.add_argument(
        "--no-affine-scale",
        action="store_true",
        help="Disable scale when fitting the affine part (do_scale=False).",
    )
    parser.add_argument(
        "--no-affine-rotation",
        action="store_true",
        help="Disable rotation when fitting the affine part (do_rotation=False).",
    )
    parser.add_argument(
        "--deform-amplitude",
        type=float,
        default=0.15,
        help="Generative nonlinear residual amplitude (default: 0.15).",
    )
    parser.add_argument(
        "--deform-freq",
        type=float,
        nargs=2,
        default=[1.3, 1.7],
        metavar=("WX", "WY"),
        help="Generative deformation frequencies (default: 1.3 1.7).",
    )
    parser.add_argument(
        "--deform-seed",
        type=int,
        default=0,
        help="RNG seed for generative deformation phases (default: 0).",
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
                        help="Integration step for analytic source.")
    parser.add_argument(
        "--delta-t",
        type=float,
        default=0.1,
        help="PUMA transition step for learned sources (one sim step = delta-t).",
    )
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of simulation steps.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip interactive plotting (still saves if --save is set).",
    )
    parser.add_argument(
        "--random-map",
        action="store_true",
        help=(
            "Generate a random transport map (affine + nonlinear) instead of using "
            "the fixed --rotation-deg / --scale / --translation / --deform-freq / --deform-seed "
            "parameters.  Only --deform-amplitude and --map-seed are used."
        ),
    )
    parser.add_argument(
        "--map-seed",
        type=int,
        default=None,
        help="Integer seed for random map generation (default: non-reproducible).",
    )
