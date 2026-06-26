# 实验流程

## 阶段 1：动力学模型 + PD

先验证 PDF 动力学模型和控制目标是否一致。

```powershell
python scripts/common/01_run_pd.py --task balance
python scripts/common/01_run_pd.py --task balance_turn
python scripts/common/01_run_pd.py --task balance_turn_roll
python scripts/common/01_run_pd.py --task height
python scripts/common/01_run_pd.py --task terrain
```

平衡抗外力测试：

```powershell
python scripts/balance/test_external_push.py --push-force 30 --push-duration 0.12 --steps 1200 --Tp-limit 1.5 --x-limit 2.0
```

## 阶段 2：神经网络模型 + MPC

采集平衡数据时建议加入随机短推扰，让模型见过外力后的恢复过程：

```powershell
python scripts/common/02_collect_data.py --task balance --episodes 120 --steps 600 --noise-scale 0.06 --push-probability 0.01 --push-force 12 --push-duration-steps 24 --T-limit 1.2 --Tp-limit 1.5
```

用 GPU 训练 NN 动力学模型：

```powershell
python scripts/common/03_train_nn.py --task balance --backend torch --device cuda --epochs 120 --hidden 256 --batch-size 4096
```

运行 NN-MPC 平衡抗外力：

```powershell
python scripts/balance/test_external_push_nn_mpc.py --backend torch --device cuda --push-force 30 --push-duration 0.12 --steps 1200 --horizon 16 --candidates 512 --Tp-limit 1.5 --x-limit 2.0
```

## 阶段 3：高斯过程模型 + MPC

GP 训练只取部分样本，适合做小数据、带不确定性惩罚的对比：

```powershell
python scripts/common/05_train_gp.py --task balance --max-points 1500
```

运行 GP-MPC 平衡抗外力：

```powershell
python scripts/balance/test_external_push_gp_mpc.py --push-force 30 --push-duration 0.12 --steps 1200 --horizon 8 --candidates 96 --uncertainty-weight 5 --Tp-limit 1.5 --x-limit 2.0
```

## 结果整理

生成汇总表：

```powershell
python scripts/balance/compare_external_push.py
```

渲染三类平衡视频：

```powershell
python scripts/balance/render_result.py --npz outputs/balance/pd/balance_external_push_30N_120ms_T1p2_Tp1p5_pd.npz --out outputs/balance/videos/01_balance_dynamics_pd.mp4 --gif outputs/balance/videos/01_balance_dynamics_pd.gif --title "Dynamics + PD, 30N push" --stride 4 --speed 0.5
python scripts/balance/render_result.py --npz outputs/balance/mpc/balance_external_push_30N_120ms_T1p2_Tp1p5_nn_mpc_torch.npz --out outputs/balance/videos/02_balance_nn_mpc.mp4 --gif outputs/balance/videos/02_balance_nn_mpc.gif --title "NN + MPC, 30N push" --stride 4 --speed 0.5
python scripts/balance/render_result.py --npz outputs/balance/mpc/balance_external_push_30N_120ms_T1p2_Tp1p5_gp_mpc.npz --out outputs/balance/videos/03_balance_gp_mpc.mp4 --gif outputs/balance/videos/03_balance_gp_mpc.gif --title "GP + MPC, 30N push" --stride 4 --speed 0.5
```

最后执行归档：

```powershell
python scripts/balance/organize_outputs.py
```

## 阶段 4：带 roll 的转向

先验证 Dynamics + PD/VMC-roll：

```powershell
python scripts/turn/test_roll_pd.py --target-deg 30 --v-ref 0.15 --steps 1200
python scripts/turn/test_roll_pd.py --target-deg 30 --v-ref 0.15 --steps 1200 --disable-roll-control
python scripts/turn/compare_roll_turn.py
```

然后进入 NN-MPC 和 GP-MPC：

```powershell
python scripts/common/02_collect_data.py --task balance_turn_roll --episodes 120 --steps 600 --noise-scale 0.06
python scripts/common/03_train_nn.py --task balance_turn_roll --backend torch --device cuda --epochs 120 --hidden 256 --batch-size 4096
python scripts/turn/test_roll_nn_mpc.py --backend torch --device cuda --target-deg 30 --v-ref 0.15 --steps 1200 --horizon 12 --candidates 256
python scripts/common/05_train_gp.py --task balance_turn_roll --max-points 1500
python scripts/turn/test_roll_gp_mpc.py --target-deg 30 --v-ref 0.15 --steps 1200 --horizon 8 --candidates 96 --uncertainty-weight 5
```
