# Terrain Adaptive GP-PMPC Plan

## Goal

Terrain adaptive control is separated from known-terrain control.

Known terrain uses the terrain height difference or an explicit terrain reference before acting. Terrain adaptive does not directly rely on known terrain height difference. It updates left-right leg height difference from state feedback such as `roll`, `roll_dot`, `support_roll` metrics, and `leg_diff` response.

The control objectives are:

- keep `theta` and `phi` stable
- keep `roll` small
- keep `x` within the safety range
- adapt `leg_diff` online without divergence
- improve or match no-adaptive stability on uneven terrain

## Current Readiness

There is no standalone `terrain_adaptive` task in `WheelLeggedEnv`. The existing project already uses:

```text
WheelLeggedEnv(task="terrain") + blind adaptive wrapper
```

Existing adaptive scripts are under `scripts/terrain_adaptive/`. They use `OnlineTerrainAdaptiveController` and `BlindTerrainCostEnv`, where the controller does not read `terrain_diff` before acting.

There is no dedicated adaptive GP model:

```text
outputs/terrain_adaptive/models/terrain_adaptive_gp.joblib  missing
outputs/terrain/models/terrain_adaptive_gp.joblib           missing
outputs/terrain/models/terrain_gp.joblib                    exists
```

Stage 7.1 therefore reuses `outputs/terrain/models/terrain_gp.joblib` as `model_source=terrain_gp_fallback`. This first version validates the adaptive controller wrapper. A dedicated adaptive residual/GP model can be trained later.

## First Version Scope

- pure Python only
- `terrain_mode=left_obstacle`
- no Isaac
- no residual GP
- no jump

## Method

Use Risk-aware GP-PMPC with safety-guided action regularization plus an adaptive leg-difference wrapper.

The PMPC planner uses `BlindTerrainCostEnv` so its cost does not use terrain height difference. The adaptive guide updates an internal `leg_diff_ref` from roll feedback. In adaptive mode, the `leg_diff` chance constraint uses center `0` as a safety proxy instead of the known-terrain `-terrain_diff` target.

## Chance Constraints

Enabled:

- `theta`
- `phi`
- `x`
- `roll`
- `leg_diff`

Disabled:

- `yaw`

## Initial Parameters

```text
horizon=4
candidates=16
uncertainty_weight=5
chance_weight=200
guide_weight=50
terminal_weight=0
k_sigma=2
noise_scale=0.03
random_fraction=0.0
adaptive_gain=0.5
adaptive_limit=0.08
```

## Acceptance Metrics

- `final_reason=max_steps`
- `max_abs_roll < 0.06 rad`, or as close as possible to known-terrain closeout
- `max_abs_theta < 0.05 rad`
- `max_abs_phi < 0.05 rad`
- `max_abs_x < 1.2 m`
- `leg_diff` does not diverge
- adaptive case has smaller roll or better stability than no-adaptive case

## Stage 7.1 First Result

Command:

```powershell
python scripts/pmpc/23_run_python_terrain_adaptive_pmpc.py --steps 1200 --horizon 4 --candidates 16 --uncertainty-weight 5 --chance-weight 200 --guide-weight 50 --terminal-weight 0 --noise-scale 0.03 --random-fraction 0.0 --adaptive-gain 0.5 --adaptive-limit 0.08 --seed 0
```

Result:

```text
model_source=terrain_gp_fallback
terrain_known_to_controller=False
final_reason=max_steps
steps=1200
max_abs_roll=0.03882 rad
max_abs_support_roll=0.05645 rad
max_abs_theta=0.03009 rad
max_abs_phi=0.03000 rad
max_abs_x=0.91361 m
max_abs_leg_diff=0.02260 m
mean_plan_time_sec=0.05070 s
total_runtime_sec=61.452 s
```

The first adaptive PMPC case reaches the main stability threshold and improves body roll relative to the known-terrain GP-PMPC closeout result. Its support-roll value is higher because terrain height difference is not known before acting; leg-difference compensation is generated from feedback rather than from `-terrain_diff`.
