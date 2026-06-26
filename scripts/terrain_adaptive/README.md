# Unknown Terrain Adaptive Scripts

This folder is separate from `scripts/terrain`.

- `scripts/terrain`: terrain difference is available to the controller.
- `scripts/terrain_adaptive`: terrain difference is not available before acting.

The adaptive controller only observes robot state, especially `roll` and
`roll_dot`. It updates an internal left-right leg height reference after the
robot contacts uneven terrain.

## Control Target

Default demo:

```text
terrain_mode = left_obstacle
v_ref = 0.15 m/s
left wheel crosses a smooth 0.04 m obstacle
terrain_known_to_controller = False
```

Targets:

```text
theta -> 0
phi -> 0
x_dot -> v_ref
roll -> 0
leg_diff is adapted online from roll feedback
```

The controller does not call `terrain_diff` to decide the action. The simulator
still records `terrain_diff` for metrics only.

## Adaptive PD/VMC

```powershell
python scripts/terrain_adaptive/test_adaptive_pd.py --terrain-mode left_obstacle --v-ref 0.15 --steps 1200
```

## Adaptive NN + MPC

Reuse the terrain dynamics model already trained in `outputs/terrain/models`.
If it does not exist yet, run:

```powershell
python scripts/common/02_collect_data.py --task terrain --episodes 120 --steps 600 --noise-scale 0.06
python scripts/common/03_train_nn.py --task terrain --backend torch --device cuda --epochs 120 --hidden 256 --batch-size 4096
```

Then run adaptive NN-MPC:

```powershell
python scripts/terrain_adaptive/test_adaptive_nn_mpc.py --backend torch --device cuda --terrain-mode left_obstacle --v-ref 0.15 --steps 1200 --horizon 8 --candidates 128
```

## Adaptive GP + MPC

Reuse the terrain GP model already trained in `outputs/terrain/models`.
If it does not exist yet, run:

```powershell
python scripts/common/05_train_gp.py --task terrain --max-points 1500
```

Then run adaptive GP-MPC:

```powershell
python scripts/terrain_adaptive/test_adaptive_gp_mpc.py --terrain-mode left_obstacle --v-ref 0.15 --steps 1200
```

## Compare, Render, Organize

```powershell
python scripts/terrain_adaptive/compare_adaptive_terrain.py

python scripts/terrain/render_terrain_result.py --npz outputs/terrain_adaptive/pd/terrain_adaptive_left_obstacle_v0p15_adaptive_pd.npz --out outputs/terrain_adaptive/videos/01_terrain_adaptive_pd.mp4 --gif outputs/terrain_adaptive/videos/01_terrain_adaptive_pd.gif --stride 10 --speed 1.0

python scripts/terrain/render_terrain_result.py --npz outputs/terrain_adaptive/mpc/terrain_adaptive_left_obstacle_v0p15_adaptive_nn_mpc_torch.npz --out outputs/terrain_adaptive/videos/02_terrain_adaptive_nn_mpc.mp4 --gif outputs/terrain_adaptive/videos/02_terrain_adaptive_nn_mpc.gif --stride 10 --speed 1.0

python scripts/terrain/render_terrain_result.py --npz outputs/terrain_adaptive/mpc/terrain_adaptive_left_obstacle_v0p15_adaptive_gp_mpc.npz --out outputs/terrain_adaptive/videos/03_terrain_adaptive_gp_mpc.mp4 --gif outputs/terrain_adaptive/videos/03_terrain_adaptive_gp_mpc.gif --stride 10 --speed 1.0

python scripts/terrain_adaptive/organize_outputs.py
```

## Useful Tests

Left obstacle:

```powershell
python scripts/terrain_adaptive/test_adaptive_pd.py --terrain-mode left_obstacle
```

Right obstacle:

```powershell
python scripts/terrain_adaptive/test_adaptive_pd.py --terrain-mode right_obstacle
```

Continuous sine terrain:

```powershell
python scripts/terrain_adaptive/test_adaptive_pd.py --terrain-mode sine
```
