# Capability Scene v3 3D Story And Collision Fix

## Purpose

This stage fixes scene storytelling and visual plausibility issues in `capability_scene_v3_3d` without changing controller, dynamics, MPC, training, or task code.

The previous v3 version already had:

- 3D/2.5D panorama rendering
- closed-chain five-link wheel-legged robot visualization
- real closeout `.npz` driven task data

This stage improves why each scene action happens and avoids obvious visual penetration.

## 1. Turn Storytelling

Turn should not be shown as only a yaw arc. The scene should explain:

- there is an obstacle on the main road
- the robot turns to bypass the obstacle
- the path smoothly returns to the main road
- trajectory heading and yaw arrow are continuous

Implementation notes:

- Add a visible obstacle in the turn zone.
- Move the turn trajectory laterally around the obstacle.
- Use trajectory tangent heading as the robot display yaw.
- Smooth heading across zone transitions.
- Print a warning if heading jumps exceed a threshold.

## 2. Height Storytelling

Height tracking should show two meaningful sub-scenes:

- height-up: raise body/leg height for ground obstacle clearance
- height-down: lower body to pass under a low-clearance gate/table/beam

The height zone still uses real height GP-PMPC `.npz` data for the robot L0 trend. The added obstacles explain the task meaning visually; they do not rerun the height controller.

## 3. No Visual Penetration

This is still a presentation scene rather than a single continuous physical simulation, but it should avoid obvious visual intersections:

- the turn path should not go through the turn obstacle
- wheels should not obviously cut through height-up obstacles
- the body should not visibly pass through the height-down beam
- known/adaptive terrain props should not cover the robot body centerline

The renderer includes a lightweight clearance check and optional debug printing:

```powershell
python scripts/panorama/render_capability_scene_v3_3d_story.py --debug-clearance
```

## Outputs

```text
outputs/panorama/capability_scene/capability_scene_v3_3d_story.npz
outputs/panorama/videos/capability_scene_v3_3d_story.mp4
outputs/panorama/videos/capability_scene_v3_3d_story.gif
outputs/panorama/figures/capability_scene_v3_3d_story_snapshot.png
```

## Scope

Only panorama scene generation/rendering and documentation are touched.
