# Known Terrain GP-PMPC Closeout

## Task Setup

- task: `terrain`
- terrain type: known terrain
- terrain mode: `left_obstacle`
- obstacle height: `0.04 m`
- obstacle start: `1.0 m`
- obstacle length: `0.5 m`
- rollout length: `1200` steps
- pure Python only

This stage does not use Isaac, residual GP, terrain adaptive control, or jump.

## Readiness Baseline

The terrain GP model exists:

```text
outputs/terrain/models/terrain_gp.joblib
```

The terrain PD/VMC baseline succeeds:

```text
final_reason=max_steps
steps=1200
max_abs_roll=0.00673 rad
max_abs_support_roll=0.01355 rad
max_abs_leg_diff_error=0.00583 m
max_abs_theta=0.03000 rad
max_abs_phi=0.03000 rad
max_abs_x=0.92158 m
```

## GP-PMPC v0 Observation

The first known-terrain GP-PMPC v0 can complete the 1200-step rollout and track leg difference, but the roll peak is too large for closeout. This shows that simply running the GP-PMPC stack is not enough; terrain needs tighter roll safety shaping.

Representative v0 result:

```text
final_reason=max_steps
steps=1200
max_abs_roll=0.19853 rad
max_abs_leg_diff_error=0.02078 m
```

## Key Tuning Changes

The closeout configuration uses:

- smaller terrain sampling noise: `noise_scale=0.03`
- no fully random candidate block: `random_fraction=0.0`
- terrain-specific guide action weights:

```text
[T, Tp, Tyaw, Froll, Fheight, leg_diff_cmd]
[1, 1, 1, 5, 2, 5]
```

- roll chance planning boundary: `0.06 rad`
- `chance_weight=200`
- `guide_weight=50`

The `0.06 rad` roll boundary is a planning safety margin. It is not the environment fall boundary.

## Recommended Parameters

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
```

Recommended command:

```powershell
python scripts/pmpc/17_run_python_terrain_pmpc.py --steps 1200 --horizon 4 --candidates 16 --uncertainty-weight 5 --chance-weight 200 --guide-weight 50 --terminal-weight 0 --noise-scale 0.03 --random-fraction 0.0 --seed 0
```

## Multi-Seed Results

| seed | final_reason | steps | max_abs_leg_diff_error m | max_abs_roll rad | max_abs_support_roll rad | max_abs_theta rad | max_abs_phi rad | max_abs_x m | mean_plan_time_sec | total_runtime_sec |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | max_steps | 1200 | 0.00565 | 0.05182 | 0.01412 | 0.03009 | 0.03000 | 0.91223 | 0.05078 | 61.557 |
| 1 | max_steps | 1200 | 0.00543 | 0.04987 | 0.01358 | 0.03000 | 0.03000 | 0.91269 | 0.12211 | 147.911 |
| 2 | max_steps | 1200 | 0.00706 | 0.05266 | 0.01765 | 0.03000 | 0.03000 | 0.93645 | 0.11992 | 145.269 |

## Artifacts

- closeout figure: `outputs/terrain/figures/terrain_pmpc_closeout.png`
- closeout metrics: `outputs/terrain/metrics/terrain_pmpc_closeout.csv`
- tuning figure: `outputs/terrain/figures/terrain_pmpc_tuning.png`
- tuning metrics: `outputs/terrain/metrics/terrain_pmpc_tuning.csv`
- recommended video: `outputs/terrain/videos/04_terrain_gp_pmpc_recommended.mp4`
- recommended GIF: `outputs/terrain/videos/04_terrain_gp_pmpc_recommended.gif`

## Conclusion

Known terrain GP-PMPC can stably traverse the known left-obstacle terrain in pure Python. Leg-difference tracking reaches millimeter-level error, roll stays below `0.06 rad`, `theta` and `phi` remain stable, and `x` stays inside the safety range.

This stage is still pure Python. It has not connected Isaac, has not trained residual GP, has not implemented terrain adaptive control, and has not extended jump.

## Next Step

The next logical terrain stage is terrain adaptive GP-PMPC: unknown terrain estimation or feedback adaptation should be added after this known-terrain closeout. Residual GP and Isaac validation should come later, and jump should remain a separate final smoke extension.
