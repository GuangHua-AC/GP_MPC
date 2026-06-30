# Terrain GP-PMPC Plan

## Stage Goal

Stage 6 first targets pure Python known terrain only. The controller is allowed to know the left/right ground height difference from the nominal terrain model, uses that known terrain difference to track a leg-difference reference, and keeps the body stable:

- track `leg_diff_ref = -terrain_diff`
- keep `theta`, `phi`, and `roll` bounded
- keep `x` inside the configured safety range
- summarize action effort for comparison

## Why Not Adaptive Yet

Known terrain is the calibration step before terrain adaptation. It verifies that the GP-PMPC stack can plan with the terrain state channels, leg-difference channel, and roll safety margins when the terrain profile is already available. Unknown-terrain adaptive behavior needs an estimator or feedback adaptation layer and should be evaluated after the known-terrain baseline is clean.

## Method

Use Risk-aware GP-PMPC with safety-guided action regularization:

1. GP mean/std predicts the pure Python `terrain` dynamics.
2. PD/VMC provides the guide action, including known terrain compensation.
3. Chance penalties discourage safety-bound violations.
4. Guide regularization keeps the short-horizon random shooting search close to stable actions.

## Chance Constraints

Enabled in the first terrain PMPC pass:

- `theta`
- `phi`
- `x`
- `roll`
- `leg_diff`

For `terrain`, `leg_diff` is centered around the known-terrain target `-terrain_diff`, not zero.

Temporarily disabled:

- `yaw`

## Initial Recommended Parameters

These inherit the stable balance/height settings and can be tuned for terrain:

```powershell
uncertainty_weight=5
chance_weight=20
guide_weight=20
terminal_weight=0
k_sigma=2
horizon=8
candidates=96
```

Stage 6.1 result: known-terrain GP-PMPC v0 already runs 1200 steps, but the peak roll is still too large for closeout:

```text
final_reason=max_steps
steps=1200
max_abs_leg_diff_error=0.01491 m
max_abs_roll=0.14144 rad
max_abs_support_roll=0.03726 rad
```

Therefore Stage 6.2 remains known terrain. It does not enter terrain adaptive yet. The current target is roll/leg-diff tuning plus runtime profiling.

## Terrain Sampling And Guide Regularization

Terrain and height tasks are more sensitive to `Froll`, `Fheight`, and `leg_diff_cmd` than balance/turn. Terrain should use smaller random-shooting noise:

```text
noise_scale=0.03
random_fraction=0.0
```

Terrain also uses action-wise guide regularization, with stronger weights on:

- `Froll`
- `Fheight`
- `leg_diff_cmd`

Default terrain guide action weights:

```text
[T, Tp, Tyaw, Froll, Fheight, leg_diff_cmd]
[1.0, 1.0, 1.0, 5.0, 2.0, 5.0]
```

## Acceptance Metrics

- `final_reason=max_steps`
- `max_abs_leg_diff_error < 0.02 m`
- `max_abs_roll < 0.06 rad`
- `max_abs_theta < 0.05 rad`
- `max_abs_phi < 0.05 rad`
- `max_abs_x < 1.2 m`
- `mean_action_norm`
- `max_action_norm`
- `mean_plan_time_sec`
- `total_runtime_sec`

## Stage 6.2 Tuning Result

The best known-terrain tuning case so far is:

```powershell
python scripts/pmpc/17_run_python_terrain_pmpc.py --steps 1200 --horizon 4 --candidates 16 --uncertainty-weight 5 --chance-weight 200 --guide-weight 50 --terminal-weight 0 --noise-scale 0.03 --random-fraction 0.0 --seed 0
```

Result:

```text
final_reason=max_steps
steps=1200
max_abs_leg_diff_error=0.00565 m
max_abs_roll=0.05182 rad
max_abs_support_roll=0.01412 rad
max_abs_theta=0.03009 rad
max_abs_phi=0.03000 rad
max_abs_x=0.91223 m
mean_plan_time_sec=0.05078 s
total_runtime_sec=61.557 s
```

This reaches the known-terrain closeout threshold for roll and leg-diff tracking. The next terrain step can be closeout packaging or a carefully separated unknown-terrain adaptive stage.
