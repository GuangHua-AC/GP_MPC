# Capability Map v3 Notes

## Purpose

Capability map v3 is a paper/defense polished version of the code-generated panorama. It keeps the v2 idea and the existing experiment results, but makes the figure easier to use in a thesis chapter or presentation.

The figure summarizes the current `wheel_legged_new` task line:

```text
平衡 -> 转向 -> 变腿高 -> 已知地形 -> 地形自适应 -> 跳跃探索
```

## What Changed From v2

- Kept the same code-generated, reproducible map concept.
- Kept reading real closeout metrics from `outputs/*/metrics` and `outputs/jump/reports`.
- Reduced each task region to only 1-2 key metrics.
- Strengthened the title hierarchy and task region labels.
- Added a bottom development route arrow.
- Split `known terrain` and `terrain adaptive` visually and semantically.
- Marked jump explicitly as `Jump / smoke tests / exploratory`.
- Added separate thesis, defense, and 16:9 slide render targets.

## Output Files

```text
outputs/panorama/figures/wheel_legged_task_big_map_v3_thesis.png
outputs/panorama/figures/wheel_legged_task_big_map_v3_thesis.pdf
outputs/panorama/figures/wheel_legged_task_big_map_v3_defense.png
outputs/panorama/figures/wheel_legged_task_big_map_v3_defense.pdf
outputs/panorama/figures/wheel_legged_task_big_map_v3_slide.png
```

## Version Intent

- Thesis version: cleaner, quieter, more formal, suitable for a paper or thesis figure.
- Defense version: larger fonts and stronger contrast, suitable for projected slides.
- Slide version: 16:9 PNG for direct insertion into presentation material.

## Metric Sources

The v3 renderer reads the same closeout/result files as v2:

- Balance: `outputs/balance/metrics/balance_pmpc_closeout.csv`
- Turn: `outputs/turn/metrics/turn_pmpc_closeout.csv`
- Height: `outputs/height/metrics/height_pmpc_closeout.csv`
- Known terrain: `outputs/terrain/metrics/terrain_pmpc_closeout.csv`
- Terrain adaptive: `outputs/terrain_adaptive/metrics/terrain_adaptive_pmpc_closeout.csv`
- Jump: `outputs/jump/reports/jump_2d_xz_pitch_sweep.csv`, with 1D sweep fallback

If a metric file is missing, the renderer prints a warning and keeps the corresponding region schematic instead of stopping the whole figure generation.

## Command

```powershell
python scripts/panorama/render_capability_map_v3.py
```

Optional single-version rendering:

```powershell
python scripts/panorama/render_capability_map_v3.py --variant thesis
python scripts/panorama/render_capability_map_v3.py --variant defense
python scripts/panorama/render_capability_map_v3.py --variant slide
```

## Scope

This is a visualization-only refinement. It does not change controller, dynamics, GP model, MPC, data collection, training, or task scripts.
