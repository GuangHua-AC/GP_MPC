# Turn scripts

转向阶段分两层：

1. `balance_turn`：只跟踪航向角 `delta/yaw`，不启用 roll 动态。
2. `balance_turn_roll`：在转向基础上启用 roll 动态，用 `Froll` 抑制侧倾。

## Roll 转向验证

先跑 Dynamics + PD/VMC-roll：

```powershell
python scripts/turn/test_roll_pd.py --target-deg 30 --v-ref 0.15 --steps 1200
```

再跑关闭 roll 控制的对照：

```powershell
python scripts/turn/test_roll_pd.py --target-deg 30 --v-ref 0.15 --steps 1200 --disable-roll-control
```

当前基线结果：

```text
PD-roll:    final_reason=max_steps, max_roll≈2.18deg
PD-no-roll: final_reason=fall_roll, 70 steps fall
```

这说明 roll 控制项是必要的，而且当前动力学 + PD/VMC-roll 已经能完成 30deg 带 roll 转向。

默认 `roll_centrifugal_ff_scale=0.0`，也就是不直接用前馈抵消离心力矩，让 roll 动态先出现，再由 `Froll` 控制消除。若要复现之前“roll 几乎恒为 0”的理想前馈版本，可以显式加：

```powershell
--roll-centrifugal-ff-scale 1.0
```

## NN + MPC

采集 `balance_turn_roll` 数据：

```powershell
python scripts/common/02_collect_data.py --task balance_turn_roll --episodes 120 --steps 600 --noise-scale 0.06
```

训练 NN 动力学模型：

```powershell
python scripts/common/03_train_nn.py --task balance_turn_roll --backend torch --device cuda --epochs 120 --hidden 256 --batch-size 4096
```

运行 NN-MPC：

```powershell
python scripts/turn/test_roll_nn_mpc.py --backend torch --device cuda --target-deg 30 --v-ref 0.15 --steps 1200 --horizon 12 --candidates 256
```

## GP + MPC

训练 GP：

```powershell
python scripts/common/05_train_gp.py --task balance_turn_roll --max-points 1500
```

运行 GP-MPC：

```powershell
python scripts/turn/test_roll_gp_mpc.py --target-deg 30 --v-ref 0.15 --steps 1200 --horizon 8 --candidates 96 --uncertainty-weight 5
```

## 汇总和视频

```powershell
python scripts/turn/compare_roll_turn.py
python scripts/turn/render_roll_turn_result.py --npz outputs/turn/pd/turn_roll_target30deg_v0p15_roll00deg_pd.npz --out outputs/turn/videos/01_turn_roll_pd.mp4 --gif outputs/turn/videos/01_turn_roll_pd.gif --title "PD/VMC-roll, 30deg turn" --stride 4 --speed 0.5
python scripts/turn/organize_outputs.py
```

后续 NN-MPC 和 GP-MPC 跑完后，也用 `render_roll_turn_result.py` 渲染到：

```text
outputs/turn/videos/02_turn_roll_nn_mpc.mp4
outputs/turn/videos/03_turn_roll_gp_mpc.mp4
```
