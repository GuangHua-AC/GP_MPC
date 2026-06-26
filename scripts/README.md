# Scripts layout

脚本按任务域拆分，避免后续平衡、转向、变腿高、地形混在一起。

```text
scripts/
  common/    通用流程：PD、采集数据、训练 NN/GP、运行 NN-MPC/GP-MPC
  balance/   平衡专用：外力推扰测试、平衡汇总、视频渲染、结果归档
  turn/      转向阶段脚本
  height/    变腿高阶段脚本
  terrain/   地形自适应阶段脚本
```

通用脚本通过 `--task` 指定任务。例如：

```powershell
python scripts/common/01_run_pd.py --task balance_turn
python scripts/common/03_train_nn.py --task balance --backend torch --device cuda
```

