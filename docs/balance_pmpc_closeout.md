# Balance GP-PMPC Closeout

## 1. Experiment setup

Current balance GP-PMPC experiments are still pure Python only. They use the
nominal wheel-legged dynamics, the trained balance GP dynamics model, and the
existing PD/VMC controller as a guide policy.

Unified test condition:

- Task: balance with external push.
- Push: 30 N, 0.12 s, starts at 1.0 s.
- Steps: 1200.
- MPC horizon: 8.
- MPC candidates: 96.
- Limits: `T_limit=1.2`, `Tp_limit=1.5`, `x_limit=2.0`.
- Risk parameters: `k_sigma=2.0`, `terminal_weight=0.0`.

## 2. Original GP-MPC failure

Under the unified `horizon=8, candidates=96` condition, the original GP-MPC
baseline does not survive the 30 N push:

| Method | Steps | Final reason | Max theta | Max phi | Max x |
| --- | ---: | --- | ---: | ---: | ---: |
| Original GP-MPC Uw0 | 237 | fall_theta | 0.7983 | 0.1951 | 0.0881 |
| Original GP-MPC Uw5 | 236 | fall_theta | 0.7927 | 0.1979 | 0.0846 |

The older archived successful GP-MPC result used a different saved rollout
configuration (`horizon=6`, `candidates=64`), so it should not be treated as a
strict parity target for this `8/96` ablation.

## 3. PMPC parity

When all PMPC risk extras are disabled:

```text
uncertainty_weight=0
chance_weight=0
guide_weight=0
terminal_weight=0
```

the new PMPC matches the original mean-only GP-MPC behavior. With
`uncertainty_weight=5` and still no chance or guide term, it also matches the
original uncertainty-only GP-MPC behavior.

This confirms that the new PMPC layer preserves mean-only parity and does not
silently add hidden terminal, chance, or guide costs.

## 4. Chance-only, guide-only, chance+guide

| Method | Steps | Final reason | Max theta | Max phi | Max x |
| --- | ---: | --- | ---: | ---: | ---: |
| PMPC Uw0 Cw0 Gw0 | 237 | fall_theta | 0.7983 | 0.1951 | 0.0881 |
| PMPC Uw5 Cw0 Gw0 | 236 | fall_theta | 0.7927 | 0.1979 | 0.0846 |
| Chance-only Uw5 Cw20 Gw0 | 238 | fall_theta | 0.7918 | 0.1931 | 0.1096 |
| Guide-only Uw5 Cw0 Gw20 | 1200 | max_steps | 0.2674 | 0.1033 | 0.8747 |
| Chance+guide Uw5 Cw20 Gw20 | 1200 | max_steps | 0.2674 | 0.1033 | 0.8747 |
| Chance+guide Uw5 Cw50 Gw20 | 1200 | max_steps | 0.2674 | 0.1033 | 0.8747 |

Main observation:

- Chance-only is not enough for this short-horizon random shooting setup. The
  chance penalty becomes useful near the boundary, but by itself it does not
  recover a stable action sequence.
- Guide-only already stabilizes the rollout because it regularizes the GP-MPC
  action sequence toward the existing PD/VMC balance behavior.
- Chance+guide is the recommended formulation because it keeps explicit risk
  information while using the PD/VMC guide as a safety prior.

## 5. Multi-seed check

Recommended setting:

```text
Uw=5, Cw=20, Gw=20, Tw=0, K=2
```

Recommended command:

```powershell
python scripts/pmpc/01_run_python_balance_pmpc.py --horizon 8 --candidates 96 --uncertainty-weight 5 --chance-weight 20 --guide-weight 20 --terminal-weight 0 --push-start 1.0 --seed 0
```

| Seed | Steps | Final reason | Max theta | Max phi | Max x |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 1200 | max_steps | 0.2674 | 0.1033 | 0.8747 |
| 1 | 1200 | max_steps | 0.2670 | 0.1032 | 0.8848 |
| 2 | 1200 | max_steps | 0.2675 | 0.1033 | 0.8744 |

Seeds 0, 1, and 2 all survive the full 1200 steps.

## 6. Recommended naming

English name:

```text
Risk-aware GP-PMPC with safety-guided action regularization
```

Chinese name:

```text
带安全引导动作正则项的风险感知 GP-PMPC
```

## 7. Current scope

- This stage is still pure Python.
- Isaac has not been connected.
- Residual GP has not been implemented.
- The guide action comes from the existing PD/VMC baseline.
- Turn, height, and terrain PMPC extensions should be done after the balance
  GP-PMPC closeout remains stable and documented.

Strict balance-stage conclusion:

```text
In the current balance external-push experiment, chance-only does not recover
stability by itself; guide-only recovers stability; chance+guide remains stable.
Therefore this stage should be described as:
Risk-aware GP-PMPC with safety-guided action regularization,
not as plain chance-constrained GP-MPC.
```

Recommended extension order:

```text
First extend turn, then height, then terrain.
Isaac validation, residual GP, and jump integration should stay later.
```

## 8. Generated artifacts

- CSV: `outputs/balance/metrics/balance_pmpc_ablation.csv`
- Figure: `outputs/balance/figures/balance_pmpc_ablation.png`
- Ablation NPZ files: `outputs/balance/pmpc/ablation/`

Closeout package artifacts:

- Recommended video MP4: `outputs/balance/videos/04_balance_gp_pmpc_recommended.mp4`
- Recommended video GIF: `outputs/balance/videos/04_balance_gp_pmpc_recommended.gif`
- Closeout figure: `outputs/balance/figures/balance_pmpc_closeout.png`
- Closeout CSV: `outputs/balance/metrics/balance_pmpc_closeout.csv`

Packaging commands:

```powershell
python scripts/pmpc/05_render_balance_pmpc_recommended.py
python scripts/pmpc/06_plot_balance_pmpc_closeout.py
```
