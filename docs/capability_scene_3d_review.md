# Capability Scene 3D Review

## Existing Panorama 3D Style

The old panorama renderer is:

```text
scripts/panorama/render_panorama_demo.py
```

Its 3D feeling comes from real Matplotlib 3D rendering:

- `fig.add_subplot(..., projection="3d")`
- `ax3d.set_proj_type("ortho")`
- `ax3d.view_init(elev=25, azim=-45)`
- a moving camera window using per-frame `xlim`, `ylim`, and `zlim`
- a 3D wireframe ground grid
- 3D boxes/cylinders for the body, wheels, obstacles, and terrain objects

So the old panorama is not just a flat 2D drawing. It is a lightweight 3D/2.5D visualization using Matplotlib 3D projection.

## Existing Robot Drawing

The old renderer already has a useful closed-chain visual leg drawing:

```text
PanoramaRenderer.draw_five_bar_leg(...)
```

It draws:

- body box
- two wheels
- motor cylinders
- upper/lower link segments
- a closed-chain five-bar-like visual structure
- wheel/body yaw, roll, and pitch through rotation matrices

Reusable functions come from:

```text
scripts/terrain/render_terrain_result.py
```

Key reusable functions/classes:

- `RenderParams`
- `rotation_matrices`
- `draw_box`
- `draw_cylinder`
- `solve_two_link_ik`

## Why capability_scene_v2 Looked Too 2D

`capability_scene_v2` kept the correct data-driven idea, but the renderer intentionally used a simple 2D axes:

- no `projection="3d"`
- no perspective or orthographic 3D camera
- terrain was drawn as a side-view filled profile
- the robot was simplified to a rectangle, two wheels, and two support legs
- no closed-chain five-link leg visualization

That made it clear and fast, but visually it did not match the old panorama style or the desired wheel-legged robot structure.

## What To Preserve

The new `capability_scene_v3_3d` should preserve:

- real closeout `.npz` driven scene data from v2
- six-zone capability timeline
- source file tracking and schematic fallback support
- zone overlays and bottom task progress

It should replace:

- the flat 2D renderer
- the simplified two-link/cart-like robot drawing

with:

- Matplotlib 3D/2.5D camera style similar to old panorama
- closed-chain five-link visual wheel-legged robot drawing
- opening panorama shot, middle follow-camera shot, and ending panorama shot

## Conclusion

The right path is not to keep patching `capability_scene_v2`, but to create a new version:

```text
capability_scene_v3_3d
```

This version should combine v2's real-task data mapping with the old panorama's 3D rendering and five-bar leg visualization.
