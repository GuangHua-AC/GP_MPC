# Capability Scene v2 Spec

## Goal

Create a reproducible, code-generated continuous scene that shows the current `wheel_legged_new` capabilities:

```text
平地平衡区 -> 转向弯道区 -> 变腿高平台区 -> 已知地形障碍区 -> 地形自适应崎岖区 -> 跳跃探索区
```

This is not an AI-generated image and not a static infographic. It is a data-driven scene that can be rendered as:

```text
outputs/panorama/videos/capability_scene_v2.mp4
outputs/panorama/videos/capability_scene_v2.gif
outputs/panorama/figures/capability_scene_v2_snapshot.png
```

## Scene Type

The first version is a data-driven unified showcase scene. It reads already successful task results and maps them into one consistent scene timeline. It does not pretend to be a single closed-loop control simulation across all tasks, because the current task references and controllers are not yet unified.

## Zones

```text
Zone 1: Balance flat zone
Zone 2: Turn curved road zone
Zone 3: Height platform zone
Zone 4: Known terrain obstacle zone
Zone 5: Terrain adaptive rough zone
Zone 6: Jump smoke zone
```

Each zone lasts about 5 s, so the full animation is about 30 s.

## Preferred Data Sources

Balance:

```text
outputs/balance/pmpc/*Uw5*Cw20*Gw20*seed0*pushStart1*gp_pmpc.npz
```

Turn:

```text
outputs/turn/pmpc/*target30deg*v0p15*Uw5*Cw20*Gw20*seed0*.npz
```

Height:

```text
outputs/height/pmpc/tracking/*step*Uw5*Cw20*Gw20*seed0*.npz
```

Known terrain:

```text
outputs/terrain/pmpc/tuning/*Cw200*Gw50*N0p03*seed0*.npz
```

Terrain adaptive:

```text
outputs/terrain_adaptive/pmpc/*Ag0p5*seed0*.npz
```

Jump:

```text
outputs/jump/npz/jump_2d_xz_pitch_smoke.npz
```

If a file is missing, the scene generator should use a schematic fallback and record `source_file=missing`.

## Unified Scene NPZ

The generated scene is saved to:

```text
outputs/panorama/capability_scene/capability_scene_v2.npz
```

It stores at least:

```text
time
zone_id
zone_name
scene_x
scene_y
robot_theta
robot_phi
robot_yaw
robot_roll
robot_L0
robot_leg_diff
action_norm
metric_texts
terrain_profile
event_flags
source_task
source_file
```

Additional arrays may be saved for rendering convenience, such as `robot_z`, `L0_ref`, `terrain_x`, and `terrain_z`.

## Rendering Style

- Render a single continuous 2.5D panorama, not six subplots.
- The robot moves left to right.
- Terrain changes from flat to curved road, height platforms, known terrain obstacle, rough terrain, and jump gap.
- The active zone label, task objective, key metrics, and source status are shown as overlays.
- The bottom progress bar shows:

```text
Balance -> Turn -> Height -> Known Terrain -> Adaptive Terrain -> Jump
```

## Scope

This scene is visualization-only. It does not change controller, dynamics, training, GP model, MPC, or task-specific scripts.
