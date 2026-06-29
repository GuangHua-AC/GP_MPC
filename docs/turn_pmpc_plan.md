# Turn GP-PMPC Plan

## 1. Goal

Stage 4 extends the balance GP-PMPC method to pure Python turn control.

Target setup:

- Task: `balance_turn_roll`.
- Yaw target: `yaw_ref = 30 deg`.
- Forward speed: `v_ref = 0.15 m/s`.
- Roll should be suppressed around `roll_ref = 0`.
- Pitch/balance states `theta` and `phi` must remain inside safety limits.
- The robot should complete the run without `fall_theta`, `fall_phi`,
  `fall_roll`, `x_out`, or `yaw_out`.

## 2. Why Not Goal-Point Navigation Yet

The first turn GP-PMPC stage only tracks a yaw reference. It does not solve
target-point navigation or path following. This keeps the experiment focused on
the turning dynamics:

- yaw tracking,
- roll suppression,
- balance safety,
- action regularization against the PD/VMC guide.

Goal-point navigation adds route planning and stopping behavior, which should be
introduced after yaw tracking is stable and well documented.

## 3. Control Method

The turn stage uses the same balance closeout method:

```text
Risk-aware GP-PMPC with safety-guided action regularization
```

The GP model predicts pure Python state14/action6 dynamics. Random-shooting
PMPC evaluates tracking cost, GP uncertainty cost, chance safety cost, optional
terminal cost, and the safety-guided action regularization term.

## 4. Chance Constraints

First enabled chance-constraint dimensions:

- `theta`
- `phi`
- `roll`
- `x`

Yaw chance constraints are not enabled in the first version. Yaw is a tracking
target, not a safety state around zero; treating yaw as a zero-centered chance
constraint would fight the commanded turn.

## 5. Initial Parameters

The first turn GP-PMPC run inherits the balance closeout parameters:

```text
uncertainty_weight = 5
chance_weight = 20
guide_weight = 20
terminal_weight = 0
k_sigma = 2
horizon = 8
candidates = 96
```

Default command:

```powershell
python scripts/pmpc/07_run_python_turn_pmpc.py --target-deg 30 --v-ref 0.15 --horizon 8 --candidates 96 --uncertainty-weight 5 --chance-weight 20 --guide-weight 20 --terminal-weight 0 --seed 0
```

## 6. Readiness Baselines

Existing baseline checks:

- PD/VMC-roll should survive 1200 steps.
- PD/VMC without roll control should fail by `fall_roll`.
- `outputs/turn/models/balance_turn_roll_gp.joblib` should exist before PMPC
  runs.

## 7. Acceptance Metrics

The comparison table should report:

- `final_reason`
- `steps`
- final yaw error in degrees
- `max_abs_roll`
- `max_abs_theta`
- `max_abs_phi`
- `max_abs_x`
- mean and max action norm

Initial acceptance target:

- `final_reason=max_steps`
- final yaw error less than about 3 deg
- roll remains far below the roll fall limit
- theta and phi remain safely bounded
- x remains inside the configured `x_limit`
