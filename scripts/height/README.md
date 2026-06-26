# Height Scripts

变腿高阶段使用 `height` 任务，输出统一放在 `outputs/height`。

## 控制目标

height 阶段包含两类展示任务：

```text
1. 离散 step_cycle:
   一边走，一边先升高一次，再降低一次。
   0.00s - 1.50s: L0_ref = 0.28 m
   1.50s - 3.00s: L0_ref = 0.36 m
   3.00s - 6.00s: L0_ref = 0.28 m

2. 连续 sine:
   一边走，一边连续上下变腿高。
   L0_ref = 0.32 + 0.04 * sin(2*pi*0.30*t)
   L0_ref range = 0.28 m ... 0.36 m
```

两类任务都使用：

```text
v_ref = 0.15 m/s
theta -> 0
phi -> 0
x_dot -> v_ref
x 只作为安全边界，不强制回原点
L0 -> L0_ref
L0_dot -> L0_dot_ref
```

控制输入中与 height 直接相关的是 `Fheight`，也就是虚拟腿方向支撑力。PD/VMC 控制律为：

```text
Fheight = F_ff + kp_L0 * (L0_ref - L0) + kd_L0 * (L0_dot_ref - L0_dot)
```

## Dynamics + PD/VMC

```powershell
python scripts/height/test_height_pd.py --mode step_cycle --low 0.28 --high 0.36 --v-ref 0.15 --steps 1200
python scripts/height/test_height_pd.py --mode sine --low 0.28 --high 0.36 --v-ref 0.15 --steps 1200
```

## NN + MPC

```powershell
python scripts/common/02_collect_data.py --task height --episodes 120 --steps 600 --noise-scale 0.06
python scripts/common/03_train_nn.py --task height --backend torch --device cuda --epochs 120 --hidden 256 --batch-size 4096

python scripts/height/test_height_nn_mpc.py --backend torch --device cuda --mode step_cycle --low 0.28 --high 0.36 --v-ref 0.15 --steps 1200 --horizon 12 --candidates 256
python scripts/height/test_height_nn_mpc.py --backend torch --device cuda --mode sine --low 0.28 --high 0.36 --v-ref 0.15 --steps 1200 --horizon 12 --candidates 256
```

## GP + MPC

```powershell
python scripts/common/05_train_gp.py --task height --max-points 1500

python scripts/height/test_height_gp_mpc.py --mode step_cycle --low 0.28 --high 0.36 --v-ref 0.15 --steps 1200 --horizon 8 --candidates 96 --uncertainty-weight 5
python scripts/height/test_height_gp_mpc.py --mode sine --low 0.28 --high 0.36 --v-ref 0.15 --steps 1200 --horizon 8 --candidates 96 --uncertainty-weight 5
```

## 对比与视频

```powershell
python scripts/height/compare_height.py

python scripts/height/render_height_result.py --npz outputs/height/pd/height_step_cycle_L0p28_to_0p36_v0p15_pd.npz --out outputs/height/videos/01_height_step_pd.mp4 --gif outputs/height/videos/01_height_step_pd.gif --stride 10 --speed 1.0
python scripts/height/render_height_result.py --npz outputs/height/mpc/height_step_cycle_L0p28_to_0p36_v0p15_nn_mpc_torch.npz --out outputs/height/videos/02_height_step_nn_mpc.mp4 --gif outputs/height/videos/02_height_step_nn_mpc.gif --stride 10 --speed 1.0
python scripts/height/render_height_result.py --npz outputs/height/mpc/height_step_cycle_L0p28_to_0p36_v0p15_gp_mpc.npz --out outputs/height/videos/03_height_step_gp_mpc.mp4 --gif outputs/height/videos/03_height_step_gp_mpc.gif --stride 10 --speed 1.0

python scripts/height/render_height_result.py --npz outputs/height/pd/height_sine_L0p28_to_0p36_v0p15_pd.npz --out outputs/height/videos/04_height_sine_pd.mp4 --gif outputs/height/videos/04_height_sine_pd.gif --stride 10 --speed 1.0
python scripts/height/render_height_result.py --npz outputs/height/mpc/height_sine_L0p28_to_0p36_v0p15_nn_mpc_torch.npz --out outputs/height/videos/05_height_sine_nn_mpc.mp4 --gif outputs/height/videos/05_height_sine_nn_mpc.gif --stride 10 --speed 1.0
python scripts/height/render_height_result.py --npz outputs/height/mpc/height_sine_L0p28_to_0p36_v0p15_gp_mpc.npz --out outputs/height/videos/06_height_sine_gp_mpc.mp4 --gif outputs/height/videos/06_height_sine_gp_mpc.gif --stride 10 --speed 1.0

python scripts/height/organize_outputs.py
```
