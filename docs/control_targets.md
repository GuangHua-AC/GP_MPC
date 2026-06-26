# 控制目标

本文档说明当前工程里每个任务的控制目标。所有任务共用 14 维状态：

```text
[theta, theta_dot, x, x_dot, phi, phi_dot,
 delta, delta_dot, psi, psi_dot,
 alpha, alpha_dot, leg_diff, leg_diff_dot]
```

动作统一为 6 维：

```text
[T, Tp, Tyaw, Froll, Fheight, leg_diff_cmd]
```

其中 `T` 是左右轮平均驱动力矩，`Tp` 是髋关节俯仰力矩，`Tyaw` 是左右轮差动力矩，`Froll` 是左右支撑力差，`Fheight` 是虚拟腿支撑力，`leg_diff_cmd` 是左右腿高差命令。

## 1. 平衡 balance

平衡任务的目标不是把机器人强行拉回原点，而是：

```text
theta -> 0
theta_dot -> 0
phi -> 0
phi_dot -> 0
x_dot -> v_ref
abs(x - x_ref) <= x_limit
```

也就是说，外力推一下以后，机器人可以向前或向后移动一段距离，只要在安全位置范围内重新站稳即可。`x` 是安全边界，不是主要跟踪目标。

默认安全边界：

```text
abs(theta) <= 0.8 rad
abs(phi)   <= 0.8 rad
abs(x)     <= 2.0 m
```

PD 控制形式：

```text
T  = kp_theta * theta + kd_theta * theta_dot + kd_x * (x_dot - v_ref)
Tp = -kp_phi * phi - kd_phi * phi_dot
```

注意：当前 PD 不再使用 `kp_x * (x - x_ref)`，避免外力推扰后机器人为了回原点而出现“先向前、再倒车”的观感。

## 2. 转向 balance_turn

在平衡基础上增加航向角跟踪：

```text
delta -> delta_ref
delta_dot -> 0
theta, phi 保持平衡
x_dot -> v_ref
```

PD 控制形式：

```text
Tyaw = kp_yaw * (delta_ref - delta) + kd_yaw * (0 - delta_dot)
```

这个阶段先验证低动态转向，即主要看航向角能否稳定跟踪，暂时不强调 roll 动态。

## 3. 带 roll 的转向 balance_turn_roll

在转向基础上加入横滚角稳定：

```text
psi -> psi_ref
psi_dot -> 0
delta -> delta_ref
theta, phi 保持平衡
```

控制里加入重力矩和离心项补偿：

```text
Froll = 2 / D * (
    -kp_roll * (psi - psi_ref)
    -kd_roll * psi_dot
    - gravity_ff_scale * gravity_roll_torque
    - centrifugal_ff_scale * centrifugal_roll_torque
)
```

这里主要对应 PDF 里的横滚与左右支撑力差关系。当前默认设置是：

```text
gravity_ff_scale = 1.0
centrifugal_ff_scale = 0.0
```

这样转向时离心力矩会先真实激发 roll，再由 `kp_roll/kd_roll` 把 roll 误差消除。若把 `centrifugal_ff_scale` 设成 `1.0`，会变成理想模型前馈，roll 会被瞬时抵消，视频里几乎看不到侧倾变化。

## 4. 变腿高 height

目标是让虚拟腿长跟踪给定高度：

```text
L0 -> L0_ref
L0_dot -> L0_dot_ref
theta, phi 保持平衡
```

控制形式：

```text
Fheight = F_ff + kp_L * (L0_ref - L0) + kd_L * (L0_dot_ref - L0_dot)
F_ff = (M + mp * eta) * g
```

## 5. 地形自适应 terrain

地形自适应先使用左右地面高度差作为观测：

```text
terrain_diff = left_ground_height - right_ground_height
```

目标是：

```text
left_leg_height - right_leg_height -> -terrain_diff
support_roll -> 0
psi -> 0
```

工程控制形式：

```text
psi_ref = atan2(terrain_diff + leg_diff, D)
leg_diff_cmd = -terrain_diff - k_roll * psi - d_roll * psi_dot
```

后续也可以把 `leg_diff_cmd` 纳入 NN-MPC 或 GP-MPC 的优化动作，让模型预测控制自己选择腿高差动作。
