# Isaac Bridge Plan

## Role of Wheel-Legged-Gym

`Wheel-Legged-Gym` should be used as:

```text
Isaac data source
+ Isaac validation platform
+ PPO / VMC baseline
```

It should not replace `wheel_legged_new`. The algorithm repository remains `wheel_legged_new`, where the physics-guided probabilistic MPC framework, GP models, PMPC logic, reports, and thesis experiments are organized.

`Wheel-Legged-Gym` is useful because it provides:

- Isaac Gym contact simulation.
- Flat and terrain environments.
- VMC-style low-level control.
- PPO training and policy baselines.
- Domain randomization options.
- More realistic friction, contact, mass, inertia, and actuator effects than the pure Python model.

## First Isaac Stage: Flat Transition Export Only

The first Isaac stage should only export offline transitions. It should not run Isaac closed-loop PMPC.

Recommended first target:

```text
wheel_legged_vmc_flat
```

Reason:

- Flat terrain is easier to debug.
- Lower terrain/contact complexity.
- Lower risk of confusing controller bugs with terrain bugs.
- Better first step for checking state/action adapters.

The planned export script is:

```text
scripts/isaac_bridge/00_export_isaac_transitions.py
```

The script should:

```text
1. Load the Isaac VMC flat task.
2. Run a small number of environments first.
3. Use VMC/PPO/random perturbation actions.
4. Extract reduced_state14 with StateAdapter.
5. Save transitions to an npz file.
```

Do not start with:

```text
rough terrain
random terrain
jump
large parallel env count
closed-loop PMPC
```

## Isaac Dataset Format

The exported Isaac dataset should contain:

```text
states
actions
next_states
dones
commands
infos
```

Expected meaning:

```text
states       reduced_state14 at time t
actions      PMPC action or Isaac VMC action, with metadata specifying which
next_states  reduced_state14 at time t+1
dones        environment termination flags
commands     velocity, yaw, height, or terrain commands used by the rollout
infos        terrain/contact/domain-randomization/debug information
```

The first validation script should check:

```text
states shape
actions shape
next_states shape
dones shape
no NaN
reasonable state ranges
reasonable action ranges
done ratio
```

Planned checker:

```text
scripts/isaac_bridge/01_check_isaac_dataset.py
```

## Residual GP Target

The final residual GP target should be:

```text
residual = isaac_next_state - nominal_python_next_state
```

Then the prediction model becomes:

```text
predicted_next_state = nominal_python_next_state + residual_gp(state, action)
```

This is the preferred thesis formulation because it uses the PDF / pure Python model as a physics prior and learns only the mismatch caused by:

- Isaac contact.
- Friction.
- Actuator differences.
- Terrain interaction.
- Modeling simplifications.

If the nominal prediction interface is not ready, the temporary fallback is:

```text
target = isaac_next_state - isaac_state
```

This fallback is direct GP delta dynamics. It is acceptable only as a bridge step before residual GP.

## Later Isaac Closed-Loop Validation

After offline data export and residual GP validation, closed-loop Isaac PMPC can start.

Recommended order:

```text
1. Isaac flat balance
2. Isaac flat turning
3. Isaac flat height tracking
4. Isaac simple step / obstacle terrain
5. Isaac random terrain
6. Isaac small-height jump smoke
```

The first closed-loop script should be:

```text
scripts/isaac_bridge/04_run_isaac_gp_pmpc_smoke.py
```

Loop structure:

```text
for each step:
    reduced_state = StateAdapter.from_isaac(env)
    pmpc_action = RiskAwareMPC.plan(reduced_state, reference)
    isaac_action = ActionAdapter.to_isaac_vmc(pmpc_action)
    env.step(isaac_action)
    save transition and debug info
```

Start with a small number of environments. Do not start with thousands of parallel environments.

## Important Notes

Do not directly copy PPO/VMC code into `wheel_legged_new`.

Do not directly treat pure Python action as Isaac action:

```text
[T, Tp, Tyaw, Froll, Fheight, leg_diff_cmd]
```

is not the same as:

```text
[left_theta0_ref, left_l0_ref, left_wheel_vel_ref,
 right_theta0_ref, right_l0_ref, right_wheel_vel_ref]
```

Use `ActionAdapter`.

Do not treat Isaac raw observation as PMPC state. Use `StateAdapter`.

Rough terrain and jump should come last. They are more sensitive to:

- Contact force reliability.
- Foot/wheel force sensor setup.
- Time step mismatch.
- Landing impact.
- Terrain mesh settings.
- Action delay and actuator limits.

The first Isaac milestone is only:

```text
flat transition export + dataset check
```
