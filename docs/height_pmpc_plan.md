# Height GP-PMPC Plan

## 1. Goal

Stage 5 extends the pure Python GP-PMPC workflow to height tracking.

Target setup:

- Task: `height`.
- Track virtual leg length `L0_ref`.
- Keep `theta` and `phi` stable.
- Keep `x` inside the configured safety range.

The first target is a fixed virtual leg length, for example `L0_ref=0.34 m`.
Step and sine references can be treated later once the fixed-height PMPC path is
stable and documented.

## 2. Scope

This stage stays pure Python. It does not connect Isaac, does not implement
residual GP, and does not extend terrain or jump. Terrain adds contact-height
uncertainty and left/right leg adaptation, which should be handled after the
basic height PMPC interface is closed out.

## 3. Method

The controller uses the same method established in balance and turn:

```text
Risk-aware GP-PMPC with safety-guided action regularization
```

The GP model predicts pure Python state14/action6 dynamics. PMPC evaluates the
nominal task cost, GP uncertainty, chance safety penalties, optional terminal
cost, and a guide-action regularization term from the existing PD/VMC baseline.

## 4. Chance Constraints

First enabled chance-constraint dimensions:

- `theta`
- `phi`
- `x`
- `L0`

Yaw, roll, and leg-difference chance constraints are not enabled for the first
height PMPC version because they are not active control objectives in the pure
height task.

`L0` is computed from the state alpha using `env.leg.L0(alpha)`. The predictive
standard deviation is approximated by mapping alpha standard deviation through
the local Jacobian `env.leg.J1(alpha)`.

## 5. Initial Parameters

The first height run inherits balance/turn PMPC parameters:

```text
uncertainty_weight = 5
chance_weight = 20
guide_weight = 20
terminal_weight = 0
k_sigma = 2
horizon = 8
candidates = 96
```

Default command:

```powershell
python scripts/pmpc/12_run_python_height_pmpc.py --L0-ref 0.34 --horizon 8 --candidates 96 --uncertainty-weight 5 --chance-weight 20 --guide-weight 20 --terminal-weight 0 --seed 0
```

## 6. Readiness Baselines

Existing readiness checks:

- `outputs/height/models/height_gp.joblib` should exist.
- Fixed-height PD/VMC should survive 1200 steps.
- The first PMPC test uses `L0_start=0.30` and `L0_ref=0.34`.

## 7. Acceptance Metrics

The comparison table should report:

- `final_reason`
- `steps`
- final L0 error
- max absolute L0 error
- max absolute `theta`
- max absolute `phi`
- max absolute `x`
- mean and max action norm

Initial acceptance target:

- `final_reason=max_steps`
- L0 error remains bounded and converges near the reference
- `theta` and `phi` stay safely below fall limits
- `x` remains inside `x_limit`
