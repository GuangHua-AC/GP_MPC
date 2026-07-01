# Capability Scene Review

## Existing Panorama Scripts

The existing panorama implementation is located at:

```text
scripts/panorama/run_panorama_demo.py
scripts/panorama/render_panorama_demo.py
```

## Data Source

`run_panorama_demo.py` creates a new continuous simulation by running:

```python
WheelLeggedEnv(task="terrain", terrain_mode="panorama")
OnlineTerrainAdaptiveController(env)
```

It does not read the already closed-out per-task result files. Instead, it produces one adaptive-PD/VMC trajectory and saves:

```text
outputs/panorama/results/panorama_showcase_adaptive_pd.npz
```

## Tasks Shown

The old panorama has four phases:

```text
balance_walk
obstacle_turn
height
adaptive_terrain
```

It shows a useful continuous scene, but it does not separately represent the completed GP-PMPC closeout results for balance, turn, height, known terrain, terrain adaptive, and jump smoke.

## Strengths

- It already renders a continuous 3D-like scene rather than six independent plots.
- It contains a robot drawing with body, wheels, and five-bar style visual links.
- It includes terrain, obstacle, height-change objects, state text, and time-series panels.
- It produces an mp4 animation from a `.npz` result.

## Gaps

- It is a fresh adaptive-PD/VMC rollout, not a data-driven montage from completed task results.
- It does not use GP-PMPC recommended results.
- It does not include a separate known-terrain zone.
- It does not include jump smoke / exploratory results.
- It has four phases, while the desired capability scene has six zones.
- It is useful as an early showcase, but it is not the final unified capability scene requested for thesis/defense presentation.

## Recommendation For Capability Scene v2

Add a new data-driven scene pipeline instead of replacing the old demo:

```text
scripts/panorama/run_capability_scene_v2.py
scripts/panorama/render_capability_scene_v2.py
scripts/panorama/30_make_capability_scene_v2.py
```

The new pipeline should:

- Read completed result `.npz` files where available.
- Map each task trajectory into a shared left-to-right scene coordinate system.
- Use schematic fallback only when a source result is missing.
- Save a unified scene `.npz`.
- Render mp4/gif/snapshot from the unified scene.

This keeps the old panorama available while creating a cleaner capability scene for project presentation.
