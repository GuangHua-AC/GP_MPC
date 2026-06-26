# 平衡阶段收尾说明

## 当前控制目标

平衡任务已经改成更接近真实机器人的目标：

```text
theta -> 0
phi -> 0
x_dot -> 0
abs(x - x_ref) <= x_limit
```

也就是说，外力推扰后机器人允许向前或向后移动，不要求回到 `x = 0`。只有超过 `x_limit`、`theta`、`phi` 等安全边界时才异常终止。

默认边界：

```text
theta_limit = 0.8 rad
phi_limit   = 0.8 rad
x_limit     = 2.0 m
```

## 三类正式平衡结果

平衡阶段最终保留三类视频和对应 `.npz`：

```text
Dynamics + PD
NN + MPC
GP + MPC
```

推荐使用同一个测试条件做对比：

```text
push_force    = 30 N
push_duration = 0.12 s
steps         = 1200
T_limit       = 1.2
Tp_limit      = 1.5
x_limit       = 2.0 m
```

## 目前还能改进的点

1. MPC 目前主要用于模型预测对比，动作采样仍然偏保守，所以在强推扰下和 PD 行为会比较接近。
2. 如果希望 NN-MPC 明显超过 PD，需要把训练数据里加入更多大推扰、初始倾角扰动和更丰富的动作探索。
3. GP-MPC 的价值主要是小数据和不确定性惩罚；它不会天然更强，核心看训练覆盖和代价函数设置。
4. 当前平衡阶段可以作为转向的基础，因为姿态目标、外力恢复、边界终止和视频输出都已经统一。

## 正式复现实验命令

```powershell
python scripts/balance/test_external_push.py --push-force 30 --push-duration 0.12 --steps 1200 --Tp-limit 1.5 --x-limit 2.0
python scripts/balance/test_external_push_nn_mpc.py --backend torch --device cuda --push-force 30 --push-duration 0.12 --steps 1200 --horizon 16 --candidates 512 --Tp-limit 1.5 --x-limit 2.0
python scripts/balance/test_external_push_gp_mpc.py --push-force 30 --push-duration 0.12 --steps 1200 --horizon 8 --candidates 96 --uncertainty-weight 5 --Tp-limit 1.5 --x-limit 2.0
```

渲染视频：

```powershell
python scripts/balance/render_result.py --npz outputs/balance/pd/balance_external_push_30N_120ms_T1p2_Tp1p5_pd.npz --out outputs/balance/videos/01_balance_dynamics_pd.mp4 --gif outputs/balance/videos/01_balance_dynamics_pd.gif --title "Dynamics + PD, 30N push" --stride 4 --speed 0.5
python scripts/balance/render_result.py --npz outputs/balance/mpc/balance_external_push_30N_120ms_T1p2_Tp1p5_nn_mpc_torch.npz --out outputs/balance/videos/02_balance_nn_mpc.mp4 --gif outputs/balance/videos/02_balance_nn_mpc.gif --title "NN + MPC, 30N push" --stride 4 --speed 0.5
python scripts/balance/render_result.py --npz outputs/balance/mpc/balance_external_push_30N_120ms_T1p2_Tp1p5_gp_mpc.npz --out outputs/balance/videos/03_balance_gp_mpc.mp4 --gif outputs/balance/videos/03_balance_gp_mpc.gif --title "GP + MPC, 30N push" --stride 4 --speed 0.5
```

整理目录：

```powershell
python scripts/balance/compare_external_push.py
python scripts/balance/organize_outputs.py
```

## 转向阶段建议

先做低动态转向：

```powershell
python scripts/common/01_run_pd.py --task balance_turn --steps 1200 --yaw-deg 20 --v-ref 0.15
```

确认 `theta, phi` 不超界、`delta` 能跟踪以后，再进入：

```powershell
python scripts/common/02_collect_data.py --task balance_turn --episodes 100 --steps 600 --noise-scale 0.06
python scripts/common/03_train_nn.py --task balance_turn --backend torch --device cuda --epochs 120 --hidden 256 --batch-size 4096
python scripts/common/04_run_nn_mpc.py --task balance_turn --backend torch --device cuda --horizon 12 --candidates 256 --steps 800 --yaw-deg 20 --v-ref 0.15
python scripts/common/05_train_gp.py --task balance_turn --max-points 1500
python scripts/common/06_run_gp_mpc.py --task balance_turn --horizon 8 --candidates 96 --uncertainty-weight 5 --steps 800 --yaw-deg 20 --v-ref 0.15
```
