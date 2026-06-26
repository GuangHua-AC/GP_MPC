# PDF 动力学到代码的对应关系

## 平面平衡

PDF 的平面平衡状态是：

```text
S = [theta, theta_dot, x, x_dot, phi, phi_dot]
u = [T, Tp]
```

代码位置：

```text
src/wheel_legged/dynamics/pdf_model.py
```

PDF 中 `N, P, Nm, Pm` 都含有 `theta_ddot, x_ddot, phi_ddot`，因此平面部分不是简单显式方程。代码把 PDF 式 (1)、(7)、(14) 写成 3 个残差方程：

```text
r(q) = 0
q = [theta_ddot, x_ddot, phi_ddot]
```

对固定状态和控制量，残差对 `q` 是线性的，所以代码用 3 个基向量构造线性系统并求解。这比手工展开质量矩阵更不容易写错。

## 低动态转向

PDF 式 (20)：

```text
delta_ddot = (TL - TR) / (R * ((mw + Iw / R^2) * D + 2 * Iz / D))
```

代码函数：

```text
yaw_ddot_from_torque(Tyaw, params)
```

其中 `Tyaw = TL - TR`。

## 高动态 roll

PDF 式 (21)：

```text
Ix * psi_ddot =
    (M + 2mp) * g * h * sin(psi)
  + (M + 2mp) * v * delta_dot * h * cos(psi)
  + (PL - PR) * D / 2
  - Cpsi * psi_dot
```

代码函数：

```text
roll_ddot(...)
```

其中 `Froll = PL - PR`。

## 支撑相变腿高

PDF 式 (30)：

```text
alpha_ddot =
    (F - (M + mp * eta) * g - (M + mp * eta^2) * J2 * alpha_dot^2)
    / ((M + mp * eta^2) * J1)
```

代码函数：

```text
height_alpha_ddot(...)
```

由于 PDF 没给五连杆完整几何尺寸，工程里先提供可替换的 `VirtualLegKinematics`：

```text
L0(alpha) = L0_offset + L0_gain * sin(alpha)
J1 = dL0 / d alpha
J2 = d2L0 / d alpha2
```

之后如果你有真实五连杆几何，只需要替换这个类即可。

