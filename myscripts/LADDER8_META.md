## How the deformation map works (ladder 8)

The full transport map is **two stages in sequence**, in the same residual form as `policy_transportation`:

$$\phi(x) = \gamma(x) + \psi(\gamma(x))$$

Only **position** and **velocity** use this map; orientation and compliance are not touched.

---

### 1. Affine part — \(\gamma(x)\)

`AffineTransform2D` implements

$$\gamma(x) = M x + t$$

with constant Jacobian \(J_\gamma = M\).

- **Default / `--random-map`:** \(M\) and \(t\) are set directly (rotation × scale + translation), or sampled randomly in `generate_random_transport`.
- **Fitted mode:** \(M, t\) come from a 2D Procrustes-style fit (rotation, optional uniform scale, translation) between source and target keypoints.

This handles global rigid-ish warping: shift, rotate, scale.

---

### 2. Nonlinear part — \(\psi(z)\)

`ParametricDeltaRegressor2D` is the **shape deformation**. It does not replace the affine map; it adds a **residual displacement** evaluated **after** the affine step:

\[
\psi(z) = \begin{bmatrix} a_0 \, g_0(z) \\ a_1 \, g_1(z) \end{bmatrix}
\]

where \(z = \gamma(x)\) and the basis functions are

\[
g_0(z) = \sin(\omega_0 z_0 + \phi_0)\cos(\omega_1 z_1 + \phi_1),\quad
g_1(z) = \cos(\omega_0 z_0 + \phi_0)\sin(\omega_1 z_1 + \phi_1).
\]

- \(\omega\) — spatial frequencies (CLI `--deform-freq` or random in `[0.5, 3]`)
- \(\phi\) — random phases from `--deform-seed` / internal RNG
- \(a_0, a_1\) — amplitudes; **`--deform-amplitude` controls how strong the warp is**

So deformation is a **smooth, bounded, position-dependent ripple** in target space. Larger amplitude ⇒ more local bending; amplitude `0` ⇒ pure affine map.

In the ladder scripts the map is usually **known, not learned**: amplitudes are set directly (or could be fitted from keypoint deltas via `fit()` if you had paired data).

---

### 3. Position transport

```text
x  --gamma-->  z = Mx + t  --psi-->  delta(z)  -->  phi(x) = z + delta(z)
```

That is `transport(x)`: affine first, then add the trig residual.

---

### 4. Velocity transport (chain rule)

Velocities push forward with the Jacobian of the full map:

\[
J_\phi(x) = J_\gamma + J_\psi(\gamma(x))\, J_\gamma,\qquad
\hat{v} = J_\phi(x)\, v
\]

`transport_velocity` does exactly that. For simulation, the source DS is integrated in the source frame, then positions and velocities are mapped into the target frame with `transport` / `transport_velocity` (or pulled back with `inverse_position` / `pull_back_velocity` when starting from target initial conditions).

---

### 5. Two ways to build the map

| Mode | How |
|------|-----|
| **Fixed / generative** (`build_fitted_transport`) | You choose rotation, scale, translation, frequencies, amplitude; map is analytic and assumed known. |
| **Random test map** (`--random-map`) | `generate_random_transport` samples affine + nonlinear parameters; only `deform_amplitude` and `map_seed` matter for reproducibility. |

Neither mode learns a GP or ILWT from demonstrations in the current ladder-8 flow; the nonlinear piece is a **closed-form parametric field**, not a data-driven regressor fit at runtime.

---

### Intuition

Think of it as: **first** move the workspace with a global affine transform, **then** add a small, smooth, wavy displacement that depends on where you landed. The amplitude knob controls how “wiggly” that second layer is, while frequencies and phases set the spatial pattern of the wiggle.