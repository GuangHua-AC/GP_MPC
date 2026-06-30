# Height GP-PMPC Closeout

## 1. Task Setup

Current height GP-PMPC closeout is pure Python only.

Experiment setup:

- Task: `height`
- Tracking modes: fixed, step, sine
- Steps: `1200`
- Horizon: `8`
- Candidates: `96`
- Limits: `T_limit=1.2`, `Tp_limit=1.5`, `x_limit=2.0`
- Recommended sampling: `noise_scale=0.03`, `random_fraction=0.0`

Reference definitions:

- Fixed: `L0_ref(t)=0.34`
- Step: `L0_ref(t)=0.30` before 2.0 s, then `0.34`
- Sine: `L0_ref(t)=0.32 + 0.02 * sin(2*pi*t/4.0)`

## 2. Baseline Readiness

- The height GP model exists at `outputs/height/models/height_gp.joblib`.
- Fixed-height PD/VMC baseline runs the full 1200 steps.

PD fixed result:

| Method | Steps | Final reason | Max L0 error after 1s | RMSE after 1s | Final L0 error | Max theta | Max phi | Max x |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PD fixed | 1200 | max_steps | 0.00000 | 0.00000 | 0.00000 | 0.0300 | 0.0300 | 0.9216 |

## 3. Recommended PMPC Parameters

```text
Uw=5
Cw=20
Gw=20
Tw=0
K=2
horizon=8
candidates=96
noise_scale=0.03
random_fraction=0.0
```

Height uses smaller sampling noise than balance/turn because the height action
is more sensitive to `L0 / alpha`. The balance/turn default `noise_scale=0.25`
made height PMPC unstable, while the existing height GP-MPC setting
`noise_scale=0.03, random_fraction=0.0` produced stable tracking.

## 4. Results

| Method | Mode | Steps | Final reason | Max L0 error after 1s | RMSE after 1s | Final L0 error | Max theta | Max phi | Max x |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PMPC fixed | fixed | 1200 | max_steps | 0.01530 | 0.01263 | -0.01426 | 0.0300 | 0.0300 | 0.0552 |
| PMPC step | step | 1200 | max_steps | 0.05269 | 0.01399 | -0.01426 | 0.0300 | 0.0300 | 0.0574 |
| PMPC sine | sine | 1200 | max_steps | 0.01524 | 0.01260 | -0.01424 | 0.0300 | 0.0300 | 0.0552 |

The step case has a larger `max_abs_L0_error_after_1s` because the reference
jumps at 2.0 s, and that instant is included in the after-1-second metric.
The RMSE after 1 s remains about 1.4 cm.

## 5. Current Stage Conclusion

Height GP-PMPC can stably complete fixed, step, and sine `L0_ref` tracking in
the pure Python height task. `theta` and `phi` remain stable, and `x` stays well
inside the configured limit.

Current scope:

- Pure Python only.
- Isaac has not been connected.
- Residual GP has not been implemented.
- Terrain and jump are not extended in this stage.

## 6. Next Step

Recommended next step:

1. Enter terrain GP-PMPC.
2. Keep Isaac validation and jump integration for later.

## 7. Generated Artifacts

- Tracking NPZ files: `outputs/height/pmpc/tracking/`
- Closeout CSV: `outputs/height/metrics/height_pmpc_closeout.csv`
- Closeout figure: `outputs/height/figures/height_pmpc_closeout.png`
- Recommended step video MP4: `outputs/height/videos/04_height_gp_pmpc_step_recommended.mp4`
- Recommended step video GIF: `outputs/height/videos/04_height_gp_pmpc_step_recommended.gif`

Packaging commands:

```powershell
python scripts/pmpc/14_run_height_pmpc_tracking_set.py
python scripts/pmpc/15_plot_height_pmpc_closeout.py
python scripts/pmpc/16_render_height_pmpc_recommended.py
```
