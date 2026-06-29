# State and Action Specification

This document freezes the state and action terminology for the PMPC framework. It is a design document only. Some fields are not yet connected to controllers or Isaac.

## Current Pure Python Raw State14

The current pure Python model uses a 14-dimensional raw state:

```text
state14 = [
  theta,
  theta_dot,
  x,
  x_dot,
  phi,
  phi_dot,
  yaw,
  yaw_dot,
  roll,
  roll_dot,
  alpha,
  alpha_dot,
  leg_diff,
  leg_diff_dot
]
```

Meaning:

```text
theta          wheel/body balancing angle
theta_dot      theta angular velocity
x              forward position
x_dot          forward velocity
phi            body pitch angle
phi_dot        body pitch angular velocity
yaw            heading / turning angle
yaw_dot        yaw rate
roll           roll angle
roll_dot       roll angular velocity
alpha          current virtual leg-height related coordinate in the pure Python model
alpha_dot      alpha rate
leg_diff       left-right leg height difference or support height difference
leg_diff_dot   leg_diff rate
```

The exact interpretation of `alpha` depends on the current pure Python model. For the PMPC paper interface, it should be converted into a clearer mean leg length state.

## Proposed Reduced State14

The PMPC layer should use a reduced state with consistent semantics across pure Python and Isaac:

```text
reduced_state14 = [
  theta,
  theta_dot,
  x,
  x_dot,
  phi,
  phi_dot,
  yaw,
  yaw_dot,
  roll,
  roll_dot,
  L0_mean,
  L0_mean_dot,
  leg_diff,
  leg_diff_dot
]
```

Meaning:

```text
theta            balancing angle
theta_dot        balancing angular velocity
x                forward position
x_dot            forward velocity
phi              body pitch
phi_dot          body pitch rate
yaw              heading angle
yaw_dot          yaw rate
roll             roll angle
roll_dot         roll rate
L0_mean          average left-right leg length / body height coordinate
L0_mean_dot      average leg length rate
leg_diff         left-right leg length difference
leg_diff_dot     left-right leg length difference rate
```

This reduced state should become the common interface for:

- Pure Python PMPC.
- Isaac exported transitions.
- Residual GP training.
- Evaluation reports.

## Current Pure Python Action6

The current pure Python force-style action is:

```text
python_action6 = [
  T,
  Tp,
  Tyaw,
  Froll,
  Fheight,
  leg_diff_cmd
]
```

Meaning:

```text
T              wheel / balance driving command
Tp             body pitch or posture command
Tyaw           yaw turning command
Froll          roll correction command
Fheight        mean height / leg-length command force
leg_diff_cmd   left-right leg-difference command
```

This action is useful for pure Python smoke tests, but it should not be sent directly to Isaac.

## Proposed PMPC Action6

The upper-level PMPC action should be actuator-reference style:

```text
pmpc_action6 = [
  theta0_mean_ref,
  theta0_diff_ref,
  L0_mean_ref,
  L0_diff_ref,
  wheel_vel_mean_ref,
  wheel_vel_diff_ref
]
```

Meaning:

```text
theta0_mean_ref       average leg virtual angle / posture reference
theta0_diff_ref       left-right difference of theta0 reference
L0_mean_ref           average leg length reference
L0_diff_ref           left-right leg length difference reference
wheel_vel_mean_ref    average wheel velocity reference
wheel_vel_diff_ref    left-right wheel velocity difference reference
```

This is recommended because it is closer to the Isaac VMC action interface than the current force-style action.

## Isaac VMC Action6

The Isaac VMC action should be:

```text
isaac_vmc_action6 = [
  left_theta0_ref,
  left_l0_ref,
  left_wheel_vel_ref,
  right_theta0_ref,
  right_l0_ref,
  right_wheel_vel_ref
]
```

The PMPC action can be mapped as:

```text
left_theta0_ref      = theta0_mean_ref + 0.5 * theta0_diff_ref
right_theta0_ref     = theta0_mean_ref - 0.5 * theta0_diff_ref

left_l0_ref          = L0_mean_ref + 0.5 * L0_diff_ref
right_l0_ref         = L0_mean_ref - 0.5 * L0_diff_ref

left_wheel_vel_ref   = wheel_vel_mean_ref + 0.5 * wheel_vel_diff_ref
right_wheel_vel_ref  = wheel_vel_mean_ref - 0.5 * wheel_vel_diff_ref
```

## Why ActionAdapter Is Required

An `ActionAdapter` is required because:

- Pure Python currently uses force-style commands.
- Isaac VMC uses reference-style commands.
- PMPC should use one abstract action definition.
- Directly sending `[T, Tp, Tyaw, Froll, Fheight, leg_diff_cmd]` into Isaac would be physically inconsistent.
- The thesis needs a clear separation between high-level planning and low-level execution.

Planned mappings:

```text
pmpc_action6 -> pure_python_action6
pmpc_action6 -> isaac_vmc_action6
```

The first mapping may be approximate in early pure Python experiments. The second mapping must follow the Isaac VMC action semantics.

## Why StateAdapter Is Required

A `StateAdapter` is required because:

- Pure Python state uses `alpha`.
- The PMPC paper interface should use `L0_mean`.
- Isaac observations contain tensors such as base angular velocity, projected gravity, joint states, wheel states, commands, and previous actions.
- Residual GP needs the same reduced state from both pure Python and Isaac.

Planned mappings:

```text
pure_python_state14 -> reduced_state14
isaac_observation / env tensors -> reduced_state14
```

## Document-Only Fields for Now

These fields are currently part of the framework specification but may not yet be fully connected to all controllers:

```text
theta0_mean_ref
theta0_diff_ref
wheel_vel_mean_ref
wheel_vel_diff_ref
isaac_vmc_action6
isaac env tensor to reduced_state14
residual GP state/action dataset
```

They should be implemented after the documentation stage through `StateAdapter` and `ActionAdapter` smoke tests.
