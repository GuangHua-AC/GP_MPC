# Balance scripts

平衡阶段专用脚本：

```text
test_external_push.py          Dynamics + PD 抗外力测试
test_external_push_nn_mpc.py   NN + MPC 抗外力测试
test_external_push_gp_mpc.py   GP + MPC 抗外力测试
sweep_external_push.py         PD 推力扫参
compare_external_push.py       生成平衡抗外力汇总表
render_result.py               渲染平衡视频
organize_outputs.py            复制正式平衡产物到 outputs/balance/final
```

正式平衡收尾命令见：`docs/balance_closeout.md`。

