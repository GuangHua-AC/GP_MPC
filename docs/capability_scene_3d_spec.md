# Capability Scene v3 3D Spec

## Goal

`capability_scene_v3_3d` is a code-generated 3D/2.5D continuous task scene for thesis and defense presentation.

It is:

- not a static infographic
- not a flat 2D schematic video
- a continuous 3D/2.5D scene similar to the old panorama demo
- driven by existing closeout `.npz` results

## Scene Structure

The robot moves through six zones:

```text
Balance flat zone
Turn curved zone
Height platform zone
Known terrain obstacle zone
Terrain adaptive rough zone
Jump exploratory zone
```

Each zone lasts about 5 s. Total scene time is about 30 s.

## Data Sources

The scene should keep using real completed results:

- Balance: GP-PMPC recommended 30N push
- Turn: GP-PMPC recommended 30 deg yaw tracking
- Height: GP-PMPC step tracking
- Known terrain: tuned GP-PMPC known terrain
- Terrain adaptive: adaptive GP-PMPC recommended
- Jump: 2D x/z/pitch jump smoke

If a source is missing, the scene generator may use schematic fallback and must record the missing source in the scene `.npz`.

## Scene Data

The generator writes:

```text
outputs/panorama/capability_scene/capability_scene_v3_3d.npz
```

It keeps v2 fields:

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

It also adds 3D-oriented fields:

```text
camera_x
ground_profile
scene_pose
robot_pose_3d_like
zone_transition_flags
```

## Renderer

The renderer writes:

```text
outputs/panorama/videos/capability_scene_v3_3d.mp4
outputs/panorama/videos/capability_scene_v3_3d.gif
outputs/panorama/figures/capability_scene_v3_3d_snapshot.png
```

Rendering requirements:

- Use Matplotlib 3D projection or equivalent 2.5D perspective.
- Use an orthographic 3D camera similar to the old panorama.
- Start with a wide panorama camera for 1-2 s.
- Use a follow camera for the middle of the scene.
- End with a wide panorama camera.
- Keep a simple title, active zone overlay, key metrics, and bottom route bar.

## Robot Model

The robot drawing must be a closed-chain five-link visual wheel-legged robot, not a cart-like simplification.

It must show:

- body
- left/right wheels
- left/right five-link leg visual structure
- hip/motor joints
- upper/lower links
- wheel axle/contact points
- leg length changes
- leg difference
- pitch/roll/height effects

The renderer should reuse the existing panorama/terrain rendering geometry utilities where possible.

## Scope

This is a visualization-only layer. It must not change controller, dynamics, GP model, MPC, data collection, training, or task-specific scripts.
