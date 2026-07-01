# wheel_legged_new

双轮足机器人动力学与控制实验工程。工程按如下路线组织：

```text
PDF 动力学模型 -> PD 控制 -> NN 动力学模型 + MPC -> GP 动力学模型 + MPC
```

模型来源：`D:\xiaochen\动力学模型\blance_turn_hight.pdf`。

## 控制任务

1. `balance`：保持 `theta, phi` 姿态稳定，`x` 只作为安全边界，不要求回到原点。
2. `balance_turn`：在平衡基础上跟踪航向角 `delta_ref`。
3. `balance_turn_roll`：转向时同时抑制横滚角 `psi`。
4. `height`：跟踪虚拟腿长 `L0_ref`。
5. `terrain`：根据左右地面高度差调节左右腿高差。

## 目录

```text
src/wheel_legged/
  dynamics/      PDF 动力学、参数、仿真环境、地形
  controllers/   PD 控制器、random-shooting MPC
  models/        sklearn NN、PyTorch NN、Gaussian Process 动力学模型
  utils/         数据与路径工具
scripts/
  common/       通用流程脚本：PD、数据采集、NN/GP 训练、NN/GP-MPC
  balance/      平衡专用脚本：外力推扰、汇总、视频、归档
  turn/         转向阶段脚本
  height/       变腿高阶段脚本
  terrain/      地形自适应阶段脚本
docs/            控制目标、动力学对应关系、实验流程
data/            训练数据
outputs/
  balance/     平衡阶段输出
  turn/        转向阶段输出，包含 balance_turn 和 balance_turn_roll
  height/      变腿高阶段输出
  terrain/     地形自适应阶段输出
tests/           冒烟测试
```

平衡阶段收尾文档见：`docs/balance_closeout.md`。

## 平衡阶段推荐命令

PD 抗外力：

```powershell
python scripts/balance/test_external_push.py --push-force 30 --push-duration 0.12 --steps 1200 --Tp-limit 1.5 --x-limit 2.0
```

NN 训练和 NN-MPC：

```powershell
python scripts/common/02_collect_data.py --task balance --episodes 120 --steps 600 --noise-scale 0.06 --push-probability 0.01 --push-force 12 --push-duration-steps 24 --T-limit 1.2 --Tp-limit 1.5
python scripts/common/03_train_nn.py --task balance --backend torch --device cuda --epochs 120 --hidden 256 --batch-size 4096
python scripts/balance/test_external_push_nn_mpc.py --backend torch --device cuda --push-force 30 --push-duration 0.12 --steps 1200 --horizon 16 --candidates 512 --Tp-limit 1.5 --x-limit 2.0
```

GP 训练和 GP-MPC：

```powershell
python scripts/common/05_train_gp.py --task balance --max-points 1500
python scripts/balance/test_external_push_gp_mpc.py --push-force 30 --push-duration 0.12 --steps 1200 --horizon 8 --candidates 96 --uncertainty-weight 5 --Tp-limit 1.5 --x-limit 2.0
```

汇总与视频：

```powershell
python scripts/balance/compare_external_push.py
python scripts/balance/render_result.py --npz outputs/balance/pd/balance_external_push_30N_120ms_T1p2_Tp1p5_pd.npz --out outputs/balance/videos/01_balance_dynamics_pd.mp4 --gif outputs/balance/videos/01_balance_dynamics_pd.gif --title "Dynamics + PD, 30N push" --stride 4 --speed 0.5
python scripts/balance/render_result.py --npz outputs/balance/mpc/balance_external_push_30N_120ms_T1p2_Tp1p5_nn_mpc_torch.npz --out outputs/balance/videos/02_balance_nn_mpc.mp4 --gif outputs/balance/videos/02_balance_nn_mpc.gif --title "NN + MPC, 30N push" --stride 4 --speed 0.5
python scripts/balance/render_result.py --npz outputs/balance/mpc/balance_external_push_30N_120ms_T1p2_Tp1p5_gp_mpc.npz --out outputs/balance/videos/03_balance_gp_mpc.mp4 --gif outputs/balance/videos/03_balance_gp_mpc.gif --title "GP + MPC, 30N push" --stride 4 --speed 0.5
python scripts/balance/organize_outputs.py
```

## GP-PMPC 阶段入口

推荐的 balance GP-PMPC 参数为 `Uw=5, Cw=20, Gw=20, Tw=0, K=2`：

```powershell
python scripts/pmpc/01_run_python_balance_pmpc.py --horizon 8 --candidates 96 --uncertainty-weight 5 --chance-weight 20 --guide-weight 20 --terminal-weight 0 --push-start 1.0 --seed 0
python scripts/pmpc/06_plot_balance_pmpc_closeout.py
```

生成推荐视频：

```powershell
python scripts/pmpc/05_render_balance_pmpc_recommended.py
```

Turn GP-PMPC 推荐命令：

```powershell
python scripts/pmpc/07_run_python_turn_pmpc.py --target-deg 30 --v-ref 0.15 --horizon 8 --candidates 96 --uncertainty-weight 5 --chance-weight 20 --guide-weight 20 --terminal-weight 0 --seed 0
python scripts/pmpc/11_plot_turn_pmpc_closeout.py
python scripts/pmpc/10_render_turn_pmpc_recommended.py
```

Height GP-PMPC 推荐命令：

```powershell
python scripts/pmpc/12_run_python_height_pmpc.py --mode step --low 0.30 --high 0.34 --switch-time 2.0 --horizon 8 --candidates 96 --uncertainty-weight 5 --chance-weight 20 --guide-weight 20 --terminal-weight 0 --seed 0
python scripts/pmpc/15_plot_height_pmpc_closeout.py
python scripts/pmpc/16_render_height_pmpc_recommended.py
```

Known terrain GP-PMPC 推荐命令：

```powershell
python scripts/pmpc/17_run_python_terrain_pmpc.py --steps 1200 --horizon 4 --candidates 16 --uncertainty-weight 5 --chance-weight 200 --guide-weight 50 --terminal-weight 0 --noise-scale 0.03 --random-fraction 0.0 --seed 0
python scripts/pmpc/21_plot_terrain_pmpc_closeout.py
python scripts/pmpc/22_render_terrain_pmpc_recommended.py
```

Terrain adaptive GP-PMPC 推荐命令：

```powershell
python scripts/pmpc/23_run_python_terrain_adaptive_pmpc.py --steps 1200 --horizon 4 --candidates 16 --uncertainty-weight 5 --chance-weight 200 --guide-weight 50 --terminal-weight 0 --noise-scale 0.03 --random-fraction 0.0 --adaptive-gain 0.5 --adaptive-limit 0.08 --seed 0
python scripts/pmpc/26_plot_terrain_adaptive_pmpc_closeout.py
python scripts/pmpc/27_render_terrain_adaptive_pmpc_recommended.py
```

## Capability Scene 最终推荐入口

最终推荐的大场景视频统一使用 `capability_scene_final` 命名：

```powershell
python scripts\panorama\36_make_capability_scene_final.py
```

输出：

```text
outputs/panorama/capability_scene/capability_scene_final.npz
outputs/panorama/videos/capability_scene_final.mp4
outputs/panorama/videos/capability_scene_final.gif
outputs/panorama/figures/capability_scene_final_snapshot.png
```

清理旧 panorama 展示输出时，先 dry-run：

```powershell
python scripts\panorama\37_cleanup_panorama_outputs.py
python scripts\panorama\37_cleanup_panorama_outputs.py --apply
```

旧版本会移动到 `outputs/panorama/archive/deprecated/`。`capability_scene_final` 才是 README、论文和答辩推荐引用版本。该视频是读取各任务 closeout `.npz` 后映射到统一 3D 场景的展示视频，不是实时自动避障统一仿真。

## GP-PMPC 实验总览

汇总 `balance`、`turn`、`height`、`known terrain`、`terrain adaptive` 五个 pure Python GP-PMPC closeout 结果，用于论文实验章节总表和总览图：

```powershell
python scripts\pmpc\28_collect_pmpc_overview.py
python scripts\pmpc\29_plot_pmpc_summary.py
```

输出：

```text
outputs/pmpc_overview/metrics/pmpc_overview.csv
outputs/pmpc_overview/figures/pmpc_task_summary.png
outputs/pmpc_overview/figures/pmpc_task_summary.pdf
```

对应文档：`docs/pmpc_experiment_overview.md`。Capability scene final 仍通过以下命令生成：

```powershell
python scripts\panorama\36_make_capability_scene_final.py
```

## 转向阶段入口

平衡收尾后，下一步从 `balance_turn` 开始：

```powershell
python scripts/common/01_run_pd.py --task balance_turn --steps 1200 --yaw-deg 20 --v-ref 0.15
```

带 roll 的转向从这里开始：

```powershell
python scripts/turn/test_roll_pd.py --target-deg 30 --v-ref 0.15 --steps 1200
python scripts/turn/test_roll_pd.py --target-deg 30 --v-ref 0.15 --steps 1200 --disable-roll-control
python scripts/turn/compare_roll_turn.py
```

确认 PD/VMC-roll 稳定后，再采集 `balance_turn_roll` 数据并训练 NN/GP 模型。

当前 roll 转向默认不前馈抵消离心力矩，因此视频中会看到转向激发 roll，再由 `Froll` 抑制回去；这比理想前馈下 roll 始终为 0 更符合动力学展示。
