# Terrain Scripts

Terrain adaptation uses the `terrain` task. Results are grouped under
`outputs/terrain`.

## Control Target

Default demo:

```text
terrain_mode = left_obstacle
v_ref = 0.15 m/s
left wheel crosses a smooth 0.04 m obstacle
right wheel stays on flat ground
```

Targets:

```text
theta -> 0
phi -> 0
x_dot -> v_ref
roll -> 0
support_roll = atan2(terrain_diff + leg_diff, D) -> 0
leg_diff -> -terrain_diff
```

Definitions:

```text
terrain_diff = left_ground_height - right_ground_height
leg_diff = left_leg_height - right_leg_height
```

So when the left wheel steps onto a 0.04 m obstacle, the controller commands
approximately `leg_diff = -0.04 m`, shortening the left side relative to the
right side. This keeps the virtual support plane level and keeps body roll
near zero.

## Dynamics + PD/VMC

```powershell
python scripts/terrain/test_terrain_pd.py --terrain-mode left_obstacle --v-ref 0.15 --steps 1200
```

## NN + MPC

```powershell
python scripts/common/02_collect_data.py --task terrain --episodes 120 --steps 600 --noise-scale 0.06
python scripts/common/03_train_nn.py --task terrain --backend torch --device cuda --epochs 120 --hidden 256 --batch-size 4096
python scripts/terrain/test_terrain_nn_mpc.py --backend torch --device cuda --terrain-mode left_obstacle --v-ref 0.15 --steps 1200 --horizon 12 --candidates 256 --noise-scale 0.03 --random-fraction 0 --mpc-blend 0.20
```

## GP + MPC

```powershell
python scripts/common/05_train_gp.py --task terrain --max-points 1500
python scripts/terrain/test_terrain_gp_mpc.py --terrain-mode left_obstacle --v-ref 0.15 --steps 1200 --horizon 4 --candidates 16 --uncertainty-weight 5 --noise-scale 0.03 --random-fraction 0 --mpc-blend 0.20
```

## Compare, Render, Organize

```powershell
python scripts/terrain/compare_terrain.py

python scripts/terrain/render_terrain_result.py --npz outputs/terrain/pd/terrain_left_obstacle_v0p15_pd.npz --out outputs/terrain/videos/01_terrain_pd.mp4 --gif outputs/terrain/videos/01_terrain_pd.gif --stride 10 --speed 1.0

python scripts/terrain/render_terrain_result.py --npz outputs/terrain/mpc/terrain_left_obstacle_v0p15_nn_mpc_torch.npz --out outputs/terrain/videos/02_terrain_nn_mpc.mp4 --gif outputs/terrain/videos/02_terrain_nn_mpc.gif --stride 10 --speed 1.0

python scripts/terrain/render_terrain_result.py --npz outputs/terrain/mpc/terrain_left_obstacle_v0p15_gp_mpc.npz --out outputs/terrain/videos/03_terrain_gp_mpc.mp4 --gif outputs/terrain/videos/03_terrain_gp_mpc.gif --stride 10 --speed 1.0

python scripts/terrain/organize_outputs.py
```
