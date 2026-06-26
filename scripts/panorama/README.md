# Panorama Showcase

This folder creates a full-map demo that combines the completed stages:

```text
Balance Walk -> Turn Around Obstacle -> Height Change -> Unknown Terrain Adaptation
```

It does not replace the individual task folders. It is a presentation/demo
layer built on top of the existing dynamics and controllers.

## Control Targets

Balance zone:

```text
v_ref = 0.18 m/s
theta -> 0
phi -> 0
roll -> 0
external push is applied briefly
```

Roll-turn zone:

```text
v_ref = 0.25 m/s
yaw_ref turns away from a large obstacle, then returns to 0 deg
roll -> 0
body, wheels, and legs rotate together in the visualization
```

Height zone:

```text
v_ref = 0.25 m/s
first L0_ref = 0.37 m to pass a small high column
then L0_ref = 0.27 m to pass under a low table-like obstacle
finally L0_ref returns to 0.32 m
theta, phi, roll stay stable
```

Adaptive-terrain zone:

```text
terrain_mode = panorama
terrain is not used directly by the controller
leg_diff is adapted online from roll and roll_dot
```

The route is meant to read as:

```text
walk while balancing
see a large obstacle that cannot be crossed
turn around the obstacle
raise leg height over a narrow column
lower leg height under a table-like obstacle
enter unknown uneven terrain and adapt leg difference online
```

## Run

```powershell
python scripts/panorama/run_panorama_demo.py --steps 5200
python scripts/panorama/compare_panorama.py
```

## Render

```powershell
python scripts/panorama/render_panorama_demo.py --npz outputs/panorama/results/panorama_showcase_adaptive_pd.npz --out outputs/panorama/videos/panorama_showcase.mp4 --gif outputs/panorama/videos/panorama_showcase.gif --stride 24 --speed 1.0
```

## Organize

```powershell
python scripts/panorama/organize_outputs.py
```
