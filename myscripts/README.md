## Custom repo

Houses Dynamical System support utils

### Test ladder
Here we implement at least the test ladder.

analytic straight-line DS with known attractor;
learned DS on one straight-line demonstration;
learned DS on one simple curved non-self-intersecting trajectory;
learned DS on one author-collected trajectory;
transported analytic DS under translation and rotation;
transported analytic DS under uniform scaling;
transported analytic DS under non-uniform scaling.
keypoint-transport baseline using the policy-transportation equation
.
Lyapunov-consistent versus Lyapunov-violating demonstrations.
local compliant or fabric-mediated execution only after the task-space
tests pass.

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

# Re-simulate an existing checkpoint
python3 myscripts/ladder3a.py --simulate-only --no-plot

# Continue training (e.g. 12000 -> 24000); loads weights and resumes automatically
python3 myscripts/ladder3b.py --max-iterations 24000 --train-only --no-plot

# Restart training from scratch (ignores checkpoint)
python3 myscripts/ladder3b.py --force-train --max-iterations 24000 --no-plot
```

Checkpoints (after a successful run): `results/ladder3_1st_order_2D/ladder3_heee/0/model`
and `results/ladder3_2nd_order_2D/ladder3_heee/0/model` under the repo root.
