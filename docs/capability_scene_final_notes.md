# Capability Scene Final Notes

## What This Scene Is

`capability_scene_v3_3d_story_*` is a data-driven unified visualization scene. It is not a single continuous closed-loop simulation of one controller solving all tasks in one run.

The scene reads closeout `.npz` results from each task, then maps their states, actions, and metrics into one shared 3D panorama:

```text
balance closeout npz
turn closeout npz
height closeout npz
known terrain closeout npz
terrain adaptive closeout npz
jump smoke npz
-> unified 3D capability scene
```

## What The Obstacles Mean

The objects in the Turn and Height zones are visual context for explaining why each capability matters. They do not mean the current controller already performs real-time obstacle perception, online obstacle avoidance, or global path planning.

- Turn: the front obstacle explains why yaw tracking and roll suppression are useful. The underlying turn GP-PMPC task is yaw tracking with roll suppression, not autonomous obstacle avoidance.
- Height: the small curb and low-clearance gate explain why changing `L0_ref` is useful. The underlying height GP-PMPC task is `L0` tracking, not real-time obstacle detection.
- Known Terrain: the narrow side rail represents a known left/right terrain height difference. This zone visualizes known `dH` and `leg_diff` tracking while keeping the body approximately level.
- Terrain Adaptive: this zone represents unknown or unprovided terrain height difference. It visualizes blind feedback / adaptive `leg_diff`, not known terrain prior tracking.
- Jump: jump remains a pure Python smoke / exploratory result, not an Isaac or hardware jump controller.

## Current Interpretation

The final capability scene should be described as:

```text
A unified closeout-results visualization for wheel-legged robot capabilities.
```

It should not be described as:

```text
A full autonomous navigation simulation with online obstacle avoidance.
```

This distinction is important for reports, papers, and defense slides.

## Final Recommended Files

Generate the final recommended scene with:

```powershell
python scripts\panorama\36_make_capability_scene_final.py
```

Recommended outputs:

```text
outputs/panorama/capability_scene/capability_scene_final.npz
outputs/panorama/videos/capability_scene_final.mp4
outputs/panorama/videos/capability_scene_final.gif
outputs/panorama/figures/capability_scene_final_snapshot.png
```

Use `capability_scene_final` for README, paper figures, and defense references. Older `capability_scene_v2*`, `capability_scene_v3_3d*`, `capability_scene_v3_3d_story*`, and intermediate motion/known-fix files are transitional outputs.

## Cleanup

Preview old panorama files that would be archived:

```powershell
python scripts\panorama\37_cleanup_panorama_outputs.py
```

Apply the archive move only after confirming the dry-run list:

```powershell
python scripts\panorama\37_cleanup_panorama_outputs.py --apply
```

Deprecated panorama outputs are moved to:

```text
outputs/panorama/archive/deprecated/
```

The cleanup only targets panorama display outputs. It does not touch `outputs/balance/`, `outputs/turn/`, `outputs/height/`, `outputs/terrain/`, `outputs/terrain_adaptive/`, `data/`, or model files.
