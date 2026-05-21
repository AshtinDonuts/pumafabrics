### Step 7 Lyapunov consistency check

For the Lyapunov candidate $L$ assumed by the learner, check whether the
demonstration velocities are consistent:

$$
\nabla L(x_t)^T\dot{x}_{demo,t} < 0.
$$

For standard SEDS-like quadratic candidates, this reduces to checking whether
the demonstration monotonically decreases distance to the attractor. If a
trajectory is non-contractive, sharply curved, or moves temporarily away from
the attractor, the learner may produce a stable but inaccurate DS by
construction.

This check comes directly from the $\tau$-SEDS motivation:
transform the data only after knowing which Lyapunov structure the
demonstration violates.

### Step 8 Visual flow-field inspection

The `pumafabrics` flow-field visualization is useful, but it should include:

- demonstration trajectories overlaid on the same plot;
- rollout trajectories;
- attractor location;
- start and end points;
- vector magnitudes, not only vector directions;
- normalized and raw-coordinate views if both are used.

Visual inspection is good for catching wrong direction, attractor mismatch,
and instability. It is weak for detecting time-scale and velocity-normalization
errors unless magnitudes are shown explicitly.
