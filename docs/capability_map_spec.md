# Capability Map v2 Specification

## Goal

Generate a reproducible, code-created capability map for `wheel_legged_new`.

The map should summarize implemented wheel-legged robot tasks in one continuous scene. It is not an AI-generated image and not a collage. It can be used in a thesis, defense slides, and README material.

## Zones

The map contains six continuous zones:

1. 平衡控制 / Balance
2. 转向控制 / Turn
3. 变腿高 / Height
4. 已知地形 / Known Terrain
5. 地形自适应 / Terrain Adaptive
6. 跳跃探索 / Jump Smoke

## Data Sources

Preferred metrics:

- Balance: `outputs/balance/metrics/balance_pmpc_closeout.csv`
- Turn: `outputs/turn/metrics/turn_pmpc_closeout.csv`
- Height: `outputs/height/metrics/height_pmpc_closeout.csv`
- Known terrain: `outputs/terrain/metrics/terrain_pmpc_closeout.csv`
- Terrain adaptive: `outputs/terrain_adaptive/metrics/terrain_adaptive_pmpc_closeout.csv`
- Jump: `outputs/jump/reports/jump_2d_xz_pitch_sweep.csv`, with fallback to other jump smoke reports

If a metric file is missing, the script should draw a schematic placeholder and report the missing file in the terminal.

## Visual Style

- One continuous wide scene, not six independent subplots.
- Terrain transitions left-to-right:
  `flat platform -> curved road -> height platform -> step obstacle -> rough unknown terrain -> jump gap`.
- Each zone has a Chinese title, English subtitle, and key metrics.
- Robot drawing is consistent across zones.
- Default output aspect ratio is 21:9.
- Outputs PNG and PDF.

## Required Output

```text
outputs/panorama/figures/wheel_legged_task_big_map_v2.png
outputs/panorama/figures/wheel_legged_task_big_map_v2.pdf
```

## Script Interface

```text
--out
--pdf
--dpi
--style {clean,dark,thesis}
--show-metrics
```

Default:

```text
--style thesis
--show-metrics
```

## Notes

The jump zone is a smoke-test capability. It can use existing 1D/2D jump metrics if present, but it should remain visually distinct from the closed-loop locomotion tasks.
