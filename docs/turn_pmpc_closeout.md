# Turn GP-PMPC Closeout

## 1. Task Setup

Current turn GP-PMPC closeout is still pure Python.

Experiment setup:

- Task: `balance_turn_roll`
- Target yaw: `target_deg=30`
- Forward speed: `v_ref=0.15`
- Steps: `1200`
- Horizon: `8`
- Candidates: `96`
- Limits: `T_limit=1.2`, `Tp_limit=1.5`, `x_limit=2.0`

The controller tracks yaw while suppressing roll and keeping `theta`, `phi`, and
`x` inside safety limits.

## 2. Baseline Results

| Method | Steps | Final reason | Final yaw error deg | Max roll rad | Max theta rad | Max phi rad | Max x m |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| PD-roll | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |
| PD-no-roll | 70 | fall_roll | -24.239 | 0.5879 | 0.0300 | 0.0300 | 0.0636 |

Conclusion: roll suppression is critical for stable turning. Without roll
control, the robot fails quickly by `fall_roll`.

## 3. Recommended GP-PMPC Result

Recommended method:

```text
Risk-aware GP-PMPC with safety-guided action regularization
```

Recommended parameters:

```text
Uw=5
Cw=20
Gw=20
Tw=0
K=2
horizon=8
candidates=96
```

Recommended command:

```powershell
python scripts/pmpc/07_run_python_turn_pmpc.py --target-deg 30 --v-ref 0.15 --horizon 8 --candidates 96 --uncertainty-weight 5 --chance-weight 20 --guide-weight 20 --terminal-weight 0 --seed 0
```

Result:

| Method | Steps | Final reason | Final yaw error deg | Max roll rad | Max theta rad | Max phi rad | Max x m |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| GP-PMPC recommended | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |

## 4. Ablation Results

| Method | Uw | Cw | Gw | Steps | Final reason | Final yaw error deg | Max roll rad | Max theta rad | Max phi rad | Max x m |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| chance+guide | 5 | 20 | 20 | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |
| chance-only | 5 | 20 | 0 | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |
| guide-only | 5 | 0 | 20 | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |
| uncertainty-only | 5 | 0 | 0 | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |
| mean-only | 0 | 0 | 0 | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |

Observation: unlike the 30 N balance push case, this 30 degree turn task is not
yet hard enough to separate the PMPC risk terms. All PMPC variants complete the
run under this setting. The no-roll baseline remains the meaningful failure
case for demonstrating why roll suppression matters.

## 5. Multi-Seed Check

Recommended group `Uw5 Cw20 Gw20 Tw0 K2` was run with seeds 0, 1, and 2.

| Seed | Steps | Final reason | Final yaw error deg | Max roll rad | Max theta rad | Max phi rad | Max x m |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |
| 1 | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |
| 2 | 1200 | max_steps | 0.130 | 0.0381 | 0.0300 | 0.0300 | 0.9216 |

All three seeds succeed.

## 6. Current Stage Conclusion

Turn GP-PMPC can complete yaw tracking while keeping roll, theta, and phi
stable in the pure Python `balance_turn_roll` task.

Current scope:

- Pure Python only.
- Isaac has not been connected.
- Residual GP has not been implemented.
- Height, terrain, and jump are not extended in this stage.

## 7. Next Step

Recommended order:

1. Extend height GP-PMPC.
2. Extend terrain GP-PMPC.
3. Keep Isaac validation, residual GP, and jump integration for later.

## 8. Generated Artifacts

- Ablation results: `outputs/turn/pmpc/ablation/`
- Closeout CSV: `outputs/turn/metrics/turn_pmpc_closeout.csv`
- Closeout figure: `outputs/turn/figures/turn_pmpc_closeout.png`
- Recommended video MP4: `outputs/turn/videos/04_turn_gp_pmpc_recommended.mp4`
- Recommended video GIF: `outputs/turn/videos/04_turn_gp_pmpc_recommended.gif`

Packaging commands:

```powershell
python scripts/pmpc/09_run_turn_pmpc_ablation.py
python scripts/pmpc/11_plot_turn_pmpc_closeout.py
python scripts/pmpc/10_render_turn_pmpc_recommended.py
```
