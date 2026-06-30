# Terrain Adaptive GP-PMPC Closeout

## Task Setup

- pure Python only
- implementation: `WheelLeggedEnv(task="terrain") + blind adaptive wrapper`
- no standalone `terrain_adaptive` env task yet
- model source: `terrain_gp_fallback`
- reused model: `outputs/terrain/models/terrain_gp.joblib`
- default terrain: `left_obstacle`
- obstacle height: `0.04 m`
- obstacle start: `1.0 m`
- obstacle length: `0.5 m`
- rollout length: `1200` steps

This stage does not connect Isaac, does not train residual GP, and does not touch jump.

## Difference From Known Terrain

Known terrain GP-PMPC directly uses the known terrain height difference to shape the leg-difference target.

Terrain adaptive GP-PMPC does not directly use the known height difference before acting. It uses a blind adaptive wrapper that generates `leg_diff_adapt` from roll/state feedback. The simulator still records terrain information for evaluation, but the adaptive controller does not use `terrain_diff` as an action reference.

## Baseline Readiness

- Existing adaptive PD/VMC baseline runs 1200 steps.
- `outputs/terrain/models/terrain_gp.joblib` exists.
- No dedicated `terrain_adaptive_gp.joblib` exists yet.

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
adaptive_gain=0.5
adaptive_limit=0.08
```

Recommended command:

```powershell
python scripts/pmpc/23_run_python_terrain_adaptive_pmpc.py --steps 1200 --horizon 4 --candidates 16 --uncertainty-weight 5 --chance-weight 200 --guide-weight 50 --terminal-weight 0 --noise-scale 0.03 --random-fraction 0.0 --adaptive-gain 0.5 --adaptive-limit 0.08 --seed 0
```

## Recommended Multi-Seed Results

| seed | final_reason | steps | max_abs_roll rad | max_abs_support_roll rad | max_abs_theta rad | max_abs_phi rad | max_abs_x m | max_abs_leg_diff m | mean_plan_time_sec | total_runtime_sec |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | max_steps | 1200 | 0.03882 | 0.05645 | 0.03009 | 0.03000 | 0.91361 | 0.02260 | 0.05070 | 61.452 |
| 1 | max_steps | 1200 | 0.04248 | 0.05539 | 0.03000 | 0.03000 | 0.90199 | 0.02218 | 0.05042 | 61.119 |
| 2 | max_steps | 1200 | 0.04210 | 0.05838 | 0.03000 | 0.03000 | 0.92068 | 0.02338 | 0.05057 | 61.287 |

## Adaptive Gain Ablation

| case | adaptive_gain | adaptive_limit | final_reason | max_abs_roll rad | max_abs_support_roll rad | max_abs_leg_diff m |
|---|---:|---:|---|---:|---:|---:|
| no_adaptive | 0.0 | 0.08 | max_steps | 0.05966 | 0.02199 | 0.00880 |
| gain_0p3 | 0.3 | 0.08 | max_steps | 0.04479 | 0.04437 | 0.01776 |
| gain_0p5 | 0.5 | 0.08 | max_steps | 0.03882 | 0.05645 | 0.02260 |
| gain_0p8 | 0.8 | 0.08 | max_steps | 0.04001 | 0.07441 | 0.02982 |
| limit_0p06 | 0.5 | 0.06 | max_steps | 0.03882 | 0.05645 | 0.02260 |
| limit_0p10 | 0.5 | 0.10 | max_steps | 0.03882 | 0.05645 | 0.02260 |

The ablation shows that `adaptive_gain=0.5` gives the lowest body roll among the tested settings, while `adaptive_gain=0.8` increases support roll and leg-diff motion. The limit sweep does not change this case because the commanded adaptive leg-diff stays below both tested limits.

## Artifacts

- closeout figure: `outputs/terrain_adaptive/figures/terrain_adaptive_pmpc_closeout.png`
- closeout metrics: `outputs/terrain_adaptive/metrics/terrain_adaptive_pmpc_closeout.csv`
- recommended video: `outputs/terrain_adaptive/videos/04_terrain_adaptive_gp_pmpc_recommended.mp4`
- recommended GIF: `outputs/terrain_adaptive/videos/04_terrain_adaptive_gp_pmpc_recommended.gif`

## Conclusion

Terrain adaptive GP-PMPC can stably complete the unknown-terrain feedback setting in pure Python. Body roll stays below `0.06 rad`, `theta` and `phi` remain stable, `x` stays within the safety range, and `leg_diff` does not diverge.

Support roll is higher than known terrain, which is expected because the adaptive controller does not directly use full terrain prior information.

## Next Step

The next recommended step is an overall GP-PMPC experiment overview across balance, turn, height, known terrain, and terrain adaptive. Residual GP and Isaac bridge should come after that, and jump should remain a final smoke extension.
