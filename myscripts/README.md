## Custom repo

Houses Dynamical System support utils

### Test ladder
Here we implement at least the test ladder.

1. analytic straight-line DS with known attractor;
	- Status: `Done`
2. learned DS on one straight-line demonstration;
	- Status: `Done`
3. learned DS on one simple curved, non-self-intersecting trajectory;
	- Status: `Done`
4. learned DS on one simple curved, self-intersecting trajectory;
	- Status: `Done`
5. transported analytic DS under translation and rotation;
	- Status: `Done`
6. transported analytic DS under uniform scaling;
	- Status: `Done`
7. transported analytic DS under non-uniform scaling.
	- Status: `Done`
8. keypoint-transport baseline using the policy-transportation equation
   $\dot{\hat{x}} = J_{\phi}(x)\dot{x}$.
	- Status: `Done`
9. Lyapunov-consistent versus Lyapunov-violating demonstrations.
	- Status: `In-progress`
10. local compliant or fabric-mediated execution only after the task-space
   tests pass.
	* Status: `In-progress`

Extras.
11. transport from one curve to another
	- Status: `Done`


### 1st and 2nd order systems
For each ladder step, we set up 1st and 2nd order systems separately;
Example: ladder1a : `1st order`.
Example: ladder1b : `2nd order`.

### Step 2 — learned DS on one straight-line demonstration
Scripts `ladder2a` (1st order) and `ladder2b` (2nd order) train PUMA on a single synthetic
line in `pumafabrics/puma_adapted/datasets/ladder2_line/`, then simulate from a grid of initials.

```bash
# Create / refresh the demonstration .npy
python3 myscripts/ladder2_data.py

# Train + simulate (GPU recommended; ~12k iterations by default)
python3 myscripts/ladder2a.py --no-plot
python3 myscripts/ladder2b.py --no-plot

# Re-simulate an existing checkpoint
python3 myscripts/ladder2a.py --simulate-only --no-plot
```

Checkpoints (after a successful run): `results/ladder2_1st_order_2D/ladder2_line/0/model`
and `results/ladder2_2nd_order_2D/ladder2_line/0/model` under the repo root.

### Step 3 — learned DS on one curved non-self-intersecting trajectory
Scripts `ladder3a` (1st order) and `ladder3b` (2nd order) train PUMA on demo 0 from
`pumafabrics/puma_adapted/datasets/LASA/heee.mat` (exported to `ladder3_heee/`), then
simulate from a grid of initials.

```bash
# Create / refresh the demonstration .npy
python3 myscripts/ladder3_data.py

# Train + simulate (GPU recommended; ~12k iterations by default)
python3 myscripts/ladder3a.py --no-plot
python3 myscripts/ladder3b.py --no-plot

# During training, ladder grid PNGs (default: every 1500 iters, same as evaluator PDFs)
# -> results/ladder3_2nd_order_2D/ladder3_heee/0/images/ladder_sim_iter_*.png
python3 myscripts/ladder3b.py --train-only --no-plot
# Custom interval or disable: --n-simulate 3000  |  --n-simulate 0

# Re-simulate an existing checkpoint
python3 myscripts/ladder3a.py --simulate-only --no-plot

# Continue training (e.g. 12000 -> 24000); loads weights and resumes automatically
python3 myscripts/ladder3b.py --max-iterations 24000 --train-only --no-plot

# Restart training from scratch (ignores checkpoint)
python3 myscripts/ladder3b.py --force-train --max-iterations 24000 --no-plot
```

Checkpoints (after a successful run): `results/ladder3_1st_order_2D/ladder3_heee/0/model`
and `results/ladder3_2nd_order_2D/ladder3_heee/0/model` under the repo root.

### Step 4 — learned DS on one curved self-intersecting trajectory
Scripts `ladder4a` (1st order) and `ladder4b` (2nd order) train PUMA on episode 0 from
`pumafabrics/puma_adapted/datasets/LAIR/capricorn/` (exported to `ladder4_capricorn/`), then
simulate from a grid of initials.

```bash
# Create / refresh the demonstration .npy
python3 myscripts/ladder4_data.py

# Train + simulate (GPU recommended; ~12k iterations by default)
python3 myscripts/ladder4a.py --no-plot
python3 myscripts/ladder4b.py --no-plot

# During training, ladder grid PNGs (default: every 1500 iters, same as evaluator PDFs)
# -> results/ladder4_2nd_order_2D/ladder4_capricorn/0/images/ladder_sim_iter_*.png
python3 myscripts/ladder4b.py --train-only --no-plot
# Custom interval or disable: --n-simulate 3000  |  --n-simulate 0

# Re-simulate an existing checkpoint
python3 myscripts/ladder4a.py --simulate-only --no-plot

# Continue training (e.g. 12000 -> 24000); loads weights and resumes automatically
python3 myscripts/ladder4b.py --max-iterations 24000 --train-only --no-plot

# Restart training from scratch (ignores checkpoint)
python3 myscripts/ladder4b.py --force-train --max-iterations 24000 --no-plot
```

Checkpoints (after a successful run): `results/ladder4_1st_order_2D/ladder4_capricorn/0/model`
and `results/ladder4_2nd_order_2D/ladder4_capricorn/0/model` under the repo root.

### Step 5 — transported DS under translation and rotation
Scripts `ladder5a` (1st order) and `ladder5b` (2nd order) apply a rigid transform
`y = R x + t` to a **source** field from any completed prior step, then simulate in the
target frame via `dy/dt = R f(R^T (y - t))` (and the same pushforward for 2nd-order
accelerations).

Source options (`--source`):

| Value | Prior step | Requires |
|-------|------------|----------|
| `analytic` | Ladder 1 straight-line DS | — |
| `learned2` | Ladder 2 straight-line demo | checkpoint under `results/ladder2_*` |
| `learned3` | Ladder 3 heee curved demo | checkpoint under `results/ladder3_*` |
| `learned4` | Ladder 4 capricorn demo | checkpoint under `results/ladder4_*` |

```bash
# Analytic source (no training)
python3 myscripts/ladder5a.py --source analytic --no-plot
python3 myscripts/ladder5b.py --source analytic --no-plot

# Learned source (train ladders 2–4 first)
python3 myscripts/ladder5a.py --source learned2 --no-plot
python3 myscripts/ladder5b.py --source learned4 --translation 0.8 -0.3 --rotation-deg 30 --no-plot
```

Default transform: translation `(1.0, 0.5)`, rotation `45°`. Override with
`--translation TX TY` and `--rotation-deg DEG`. For `analytic` only, set the source
attractor with `--attractor X Y` (transported goal is `R x* + t`).

Learned sources advance one PUMA ``transition`` per simulation step (``--delta-t``,
default `0.1`, matching ladders 2–4). Analytic sources use Euler steps with ``--dt``
(default `0.01`). Use enough ``--steps`` for learned runs (e.g. `2000`–`4000`, as in
ladder 2–4).

Figures: `myscripts/images/ladder5a_transported_ds.png`,
`myscripts/images/ladder5b_transported_ds_2nd_order.png`.

### Step 6 — transported DS under uniform scaling
Scripts `ladder6a` (1st order) and `ladder6b` (2nd order) apply a uniform scale
`y = s x + t` to a **source** field from any completed prior step, then simulate in the
target frame via `dy/dt = s f((y - t) / s)` (and the same pushforward for 2nd-order
accelerations).

Source options (`--source`):

| Value | Prior step | Requires |
|-------|------------|----------|
| `analytic` | Ladder 1 straight-line DS | — |
| `learned2` | Ladder 2 straight-line demo | checkpoint under `results/ladder2_*` |
| `learned3` | Ladder 3 heee curved demo | checkpoint under `results/ladder3_*` |
| `learned4` | Ladder 4 capricorn demo | checkpoint under `results/ladder4_*` |

```bash
# Analytic source (no training)
python3 myscripts/ladder6a.py --source analytic --no-plot
python3 myscripts/ladder6b.py --source analytic --no-plot

# Learned source (train ladders 2–4 first)
python3 myscripts/ladder6a.py --source learned2 --no-plot
python3 myscripts/ladder6b.py --source learned4 --scale 1.5 --translation 0.2 0.1 --no-plot
```

Default transform: scale `2.0`, translation `(0, 0)`. Override with `--scale S` and
`--translation TX TY`. For `analytic` only, set the source attractor with
`--attractor X Y` (transported goal is `s x* + t`).

Learned sources advance one PUMA ``transition`` per simulation step (``--delta-t``,
default `0.1`, matching ladders 2–4). Analytic sources use Euler steps with ``--dt``
(default `0.01`). Use enough ``--steps`` for learned runs (e.g. `2000`–`4000`, as in
ladder 2–4).

Figures: `myscripts/images/ladder6a_transported_ds.png`,
`myscripts/images/ladder6b_transported_ds_2nd_order.png`.

### Step 7 — transported DS under non-uniform scaling
Scripts `ladder7a` (1st order) and `ladder7b` (2nd order) apply a diagonal non-uniform
scale `y = S x + t` with `S = diag(s_x, s_y)` to a **source** field from any completed
prior step, then simulate in the target frame via `dy/dt = S f(S^{-1} (y - t))` (and the
same pushforward for 2nd-order accelerations).

Source options (`--source`):

| Value | Prior step | Requires |
|-------|------------|----------|
| `analytic` | Ladder 1 straight-line DS | — |
| `learned2` | Ladder 2 straight-line demo | checkpoint under `results/ladder2_*` |
| `learned3` | Ladder 3 heee curved demo | checkpoint under `results/ladder3_*` |
| `learned4` | Ladder 4 capricorn demo | checkpoint under `results/ladder4_*` |

```bash
# Analytic source (no training)
python3 myscripts/ladder7a.py --source analytic --no-plot
python3 myscripts/ladder7b.py --source analytic --no-plot

# Learned source (train ladders 2–4 first)
python3 myscripts/ladder7a.py --source learned2 --no-plot
python3 myscripts/ladder7b.py --source learned4 --scale 1.5 0.75 --translation 0.2 0.1 --no-plot
```

Default transform: scale `(2.0, 0.5)`, translation `(0, 0)`. Override with
`--scale SX SY` and `--translation TX TY`. For `analytic` only, set the source attractor
with `--attractor X Y` (transported goal is `S x* + t`).

Learned sources advance one PUMA ``transition`` per simulation step (``--delta-t``,
default `0.1`, matching ladders 2–4). Analytic sources use Euler steps with ``--dt``
(default `0.01`). Use enough ``--steps`` for learned runs (e.g. `2000`–`4000`, as in
ladder 2–4).

Figures: `myscripts/images/ladder7a_transported_ds.png`,
`myscripts/images/ladder7b_transported_ds_2nd_order.png`.

### Step 8 — keypoint policy transport (affine + nonlinear deformation)
Scripts `ladder8a` (1st order) and `ladder8b` (2nd order) transport a **source** field from
any completed prior step using the
[policy_transportation](https://github.com/franzesegiovanni/policy_transportation) residual map

    phi(x) = gamma(x) + psi(gamma(x)),    v_hat = J_phi(x) @ v,

with `J_phi = J_gamma + J_psi @ J_gamma`. The affine part `gamma` is fit from source/target
keypoints; the nonlinear part `psi` is an RBF residual fit on `target - gamma(source)`.
Only position (`transport`) and velocity (`transport_velocity`) are transported.

Source options (`--source`):

| Value | Prior step | Requires |
|-------|------------|----------|
| `analytic` | Ladder 1 straight-line DS | — |
| `learned2` | Ladder 2 straight-line demo | checkpoint under `results/ladder2_*` |
| `learned3` | Ladder 3 heee curved demo | checkpoint under `results/ladder3_*` |
| `learned4` | Ladder 4 capricorn demo | checkpoint under `results/ladder4_*` |

```bash
# Analytic source (no training)
python3 myscripts/ladder8a.py --source analytic --no-plot
python3 myscripts/ladder8b.py --source analytic --no-plot

# Learned source (train ladders 2–4 first)
python3 myscripts/ladder8a.py --source learned2 --no-plot
python3 myscripts/ladder8b.py --source learned4 --deform-amplitude 0.1 --no-plot
```

CLI warp parameters define a **generative** target keypoint distribution used to fit
`PolicyTransportation`; defaults: translation `(0.5, 0.3)`, rotation `25°`, scale `1.2`,
deform amplitude `0.15`, frequencies `(1.3, 1.7)`, seed `0`. Override with
`--translation`, `--rotation-deg`, `--scale`, `--deform-amplitude`, `--deform-freq`,
`--deform-seed`, and `--no-affine-scale` / `--no-affine-rotation` for the affine fit.
For `analytic` only, set the source attractor with `--attractor X Y`.

Learned sources advance one PUMA ``transition`` per simulation step (``--delta-t``,
default `0.1`). Analytic sources use Euler steps with ``--dt`` (default `0.01`).

Figures: `myscripts/images/ladder8a_policy_transport_ds.png`,
`myscripts/images/ladder8b_policy_transport_ds_2nd_order.png`.

### Step 11 — transport from source DS to a real demo curve (LASA / LAIR)
Scripts `ladder11a` (1st order) and `ladder11b` (2nd order) fit a `PolicyTransportation2D`
map **directly from data**: source keypoints come from the source system's demonstration
(or a synthetic straight-line for `analytic`), and target keypoints are resampled from a
real demonstration chosen from the LASA or LAIR datasets.

The map is residual:

    phi(x) = gamma(x) + psi(gamma(x)),    v_hat = J_phi(x) @ v

`gamma` is an affine Procrustes fit; `psi` is a Gaussian-RBF interpolant fit to
`target - gamma(source)`.  A Newton-based inverse (affine warm start) is used to
map initial states from target to source frame.

Source options (`--source`): same as steps 5–8.

Target options:

| `--target-dataset` | `--target-name` examples |
|--------------------|--------------------------|
| `lasa` | `heee` `CShape` `Sine` `Snake` `Worm` `Sshape` `GShape` ... |
| `lair` | `capricorn` `e` `mountain` `phi` `double_loop` `two` ... |

```bash
# Analytic source → LASA heee (default)
python3 myscripts/ladder11a.py --source analytic --no-plot
python3 myscripts/ladder11b.py --source analytic --no-plot

# Analytic source → LASA CShape
python3 myscripts/ladder11a.py --source analytic --target-dataset lasa --target-name CShape --no-plot

# Analytic source → LAIR capricorn
python3 myscripts/ladder11a.py --source analytic --target-dataset lair --target-name capricorn --no-plot

# Learned source (train ladder 3 first) → LASA Sine
python3 myscripts/ladder11a.py --source learned3 --target-dataset lasa --target-name Sine --no-plot

# Learned source → LAIR mountain, finer resampling, explicit RBF bandwidth
python3 myscripts/ladder11b.py --source learned4 --target-dataset lair --target-name mountain \
    --n-keypoints 60 --rbf-bandwidth 0.2 --no-plot

# Disable affine scale/rotation (translation-only affine part)
python3 myscripts/ladder11a.py --no-affine-scale --no-affine-rotation --no-plot
```

Key CLI parameters beyond `--source`:

| Flag | Default | Meaning |
|------|---------|---------|
| `--target-dataset` | `lasa` | Dataset family: `lasa` or `lair` |
| `--target-name` | `heee` | Dataset name within the family |
| `--target-demo` | `0` | Demo / episode index |
| `--n-keypoints` | `40` | Points to resample both curves to before fitting |
| `--grid-nx` / `--grid-ny` | `4` / `3` | Initial-state grid inside **target** demo bbox (12 points) |
| `--grid-margin` | `0.12` | Inset inside target bbox (same idea as ladder 3) |
| `--rbf-bandwidth` | auto | Gaussian RBF bandwidth h (default: median heuristic) |
| `--rbf-reg` | `1e-6` | Tikhonov regularisation for the RBF linear solve |
| `--no-affine-scale` | — | Disable scale in affine part |
| `--no-affine-rotation` | — | Disable rotation in affine part |

Figures: `myscripts/images/ladder11a_demo_transport_ds.png`,
`myscripts/images/ladder11b_demo_transport_ds_2nd_order.png`.

---


## Version Updates:

#### current work: uncommitted
Ladder ready up till step 8, plus extra step 11 (demo-to-demo transport).

#### git head: `a36a058`:
Fixed time-step mismatch.
What was wrong: Learned fields were queried with PUMA’s transition (step size delta_t = 0.1), but integration used Euler with dt = 0.01. Each step only advanced 1/10 of what the network expects, so grid points barely moved.

#### git head: `00c1c94`
We moved the simulate and plot visualizations from `PUMA_ROOT/myscripts/` to `PUMA_ROOT/results/`.
The experiments under `results/` are organized under the name of each trajectory set.
Ladder ready up till step 3.
