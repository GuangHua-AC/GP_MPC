# Physics-Guided Probabilistic MPC Framework

## Project Goal

The long-term project goal is:

**Physics-guided Probabilistic MPC for Wheel-Legged Robot Balance, Turning, Height Tracking, Terrain Adaptation and Jump Smoke.**

The project should not be organized as a direct code transplant from another repository. The main contribution should be a clear algorithm pipeline:

```text
PDF / Pure Python nominal dynamics
-> PD / VMC baseline
-> NN-MPC
-> GP-MPC
-> GP-PMPC uncertainty-aware / chance-constrained
-> residual GP
-> Isaac validation
```

The current `wheel_legged_new` project already contains the pure Python dynamics, PD/VMC controllers, NN dynamics, GP dynamics, MPC scripts, metrics, reports, and rendered videos for balance, turning, height tracking, terrain adaptation, panorama, and jump smoke. Therefore, it should remain the algorithm and experiment repository.

## Repository Roles

### wheel_legged_new as the Algorithm Repository

`wheel_legged_new` is the main research repository because it already provides:

- Pure Python nominal dynamics based on the PDF model.
- Task-specific scripts for `balance`, `turn`, `height`, `terrain`, `terrain_adaptive`, `panorama`, and `jump`.
- PD/VMC baselines.
- NN dynamics models and NN-MPC.
- GP dynamics models and GP-MPC.
- Organized outputs, metrics, `.npz` results, and videos.
- Jump smoke tests from 1D to 2D x/z/pitch coupling.

This repository should contain the thesis algorithm:

```text
state/action abstraction
physics-guided nominal model
probabilistic dynamics
risk-aware MPC
chance constraints
evaluation scripts
figures and reports
```

### Wheel-Legged-Gym as Isaac Data Source and Validation Platform

`Wheel-Legged-Gym` should not replace the current project. It should be used as:

- Isaac Gym high-fidelity contact simulation.
- A source of transition data under contact, friction, motor, and domain randomization effects.
- A validation platform for flat balance, turning, height tracking, terrain adaptation, and later jump smoke.
- A PPO / VMC baseline for comparison.

The Isaac repository has its own environment/config/task registry structure. Its action interface is not the same as the pure Python force-style action interface in `wheel_legged_new`. Therefore, direct code copying would blur the research logic and make the control stack harder to explain.

## Overall Framework

```text
PDF / Pure Python nominal dynamics
        |
        v
PD / VMC baseline
        |
        v
Transition data: (state, action, next_state, info)
        |
        +--------------------+
        |                    |
        v                    v
NN dynamics + MPC       GP dynamics + MPC
                             |
                             v
              GP-PMPC uncertainty-aware MPC
                             |
                             v
              Chance-constrained probabilistic MPC
                             |
                             v
        Residual GP: Isaac next state - nominal prediction
                             |
                             v
                    Isaac validation
```

The final method should be physics-guided rather than purely black-box:

```text
s_next = f_nominal_pdf_python(s, u) + GP_residual(s, u)
```

The GP uncertainty should be used inside MPC for:

- Risk-aware costs.
- Safety margins.
- Chance constraints.
- Data collection / memory gating in later online variants.

## Thesis Chapter Structure

Recommended thesis structure:

```text
Chapter 1: Introduction
- Wheel-legged robot balance, turning, height tracking, terrain adaptation, and jump challenges.
- Limitations of pure analytic dynamics.
- Need for high-fidelity simulation and uncertainty-aware model predictive control.
- Proposed physics-guided probabilistic MPC framework.

Chapter 2: Robot Dynamics and Task Modeling
- PDF / analytic dynamics.
- Pure Python nominal model.
- State definition.
- Action definition.
- Task definitions for balance, turning, height, terrain, and jump smoke.

Chapter 3: PD / VMC Baseline Control
- Low-level PD/VMC design.
- Balance with external push.
- Roll-aware turning.
- Variable height tracking.
- Terrain adaptation.

Chapter 4: Learned Dynamics Models
- Transition data collection.
- NN dynamics model.
- GP dynamics model.
- Direct delta prediction.
- Residual GP formulation.
- One-step and multi-step prediction evaluation.

Chapter 5: Probabilistic MPC
- Random shooting MPC baseline.
- GP-MPC mean prediction.
- Uncertainty-aware cost.
- Diagonal covariance propagation.
- Chance constraints.
- Safety boundary design.

Chapter 6: Simulation Experiments
- Pure Python balance comparison.
- Turning comparison.
- Height tracking comparison.
- Terrain adaptation comparison.
- Jump smoke validation.
- Data efficiency and uncertainty ablation.
- Isaac validation after adapter and residual GP are ready.

Chapter 7: Conclusion and Future Work
- Summary of physics-guided PMPC.
- Limits of simplified models.
- Future Isaac and real-robot deployment.
```

## Stage Acceptance Criteria

### Stage 0: Documentation and Interface Freeze

Acceptance:

- `framework_gp_pmpc.md` exists.
- `state_action_spec.md` exists.
- `isaac_bridge_plan.md` exists.
- The project state/action definitions are explicit.
- The role of `wheel_legged_new` and `Wheel-Legged-Gym` is clear.

### Stage 1: StateAdapter and ActionAdapter

Acceptance:

- Pure Python state14 can be converted to `reduced_state14`.
- PMPC action can be mapped to pure Python action.
- PMPC action can be mapped to Isaac VMC action.
- Action bounds are checked.
- No MPC or Isaac closed-loop is required at this stage.

### Stage 2: Pure Python GP-PMPC

Acceptance:

- Balance external push can run with mean-only GP-MPC.
- Balance external push can run with uncertainty-aware GP-PMPC.
- Metrics include max pitch, max body pitch, x drift, fall reason, and uncertainty cost.
- Existing PD, NN-MPC, and GP-MPC outputs remain valid baselines.

### Stage 3: Chance Constraints

Acceptance:

- Constraints include pitch, body pitch, roll, leg height, leg difference, and action limits.
- The controller reports chance-constraint violation margins.
- Increasing uncertainty increases the penalty or margin.

### Stage 4: Residual GP

Acceptance:

- A nominal prediction function is available.
- Residual target is:

```text
residual = measured_next_state - nominal_python_next_state
```

- One-step and multi-step prediction errors are reported.
- Uncertainty calibration is evaluated.

### Stage 5: Isaac Offline Dataset

Acceptance:

- Isaac flat transitions are exported without closed-loop PMPC.
- Dataset contains `states`, `actions`, `next_states`, `dones`, `commands`, and `infos`.
- Shapes are correct.
- No NaN values.
- State ranges are physically reasonable.

### Stage 6: Isaac Validation

Acceptance:

- Start with flat balance only.
- Run a small number of environments first.
- Save `.npz`, metrics, and video.
- Do not start with rough terrain or jump.

### Stage 7: Full Comparison

Acceptance:

- Compare PD/VMC, NN-MPC, GP-MPC, GP-PMPC, chance-constrained GP-PMPC, and PPO/VMC baseline where available.
- Report success rate, fall reason, tracking error, effort, uncertainty, and constraint violations.
