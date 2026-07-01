# Capability Map Review

## Existing Implementation

Current panorama implementation is in:

- `scripts/panorama/run_panorama_demo.py`
- `scripts/panorama/render_panorama_demo.py`
- `scripts/panorama/compare_panorama.py`
- `scripts/panorama/organize_outputs.py`

Current outputs are in:

- `outputs/panorama/results/panorama_showcase_adaptive_pd.npz`
- `outputs/panorama/videos/panorama_showcase.mp4`
- `outputs/panorama/metrics/panorama_summary.csv`

A fresh archive copy was generated at:

- `outputs/panorama/archive/panorama_showcase_adaptive_pd_20260630_135652.npz`
- `outputs/panorama/archive/panorama_showcase_20260630_135652.mp4`

## What It Shows

The existing panorama script renders a dynamic full-map demo with:

- balance walk with push
- obstacle turn
- height change
- unknown terrain adaptation

It is not a simple collage. It generates one continuous simulated route and renders it as a 3D animation.

## Data Source

The old panorama reads its own generated `panorama_showcase_adaptive_pd.npz`. It does not directly read the new PMPC closeout result files for balance, turn, height, known terrain, or terrain adaptive.

## Strengths

- One continuous route instead of isolated subplots.
- Real simulation trajectory, not an AI image.
- Uses robot and terrain rendering already aligned with the project.
- Demonstrates an intuitive story: balance, turn, height, then terrain adaptation.

## Current Problems

- It is a video demo, not a static paper-ready capability map.
- It does not include known terrain GP-PMPC.
- It does not include terrain adaptive GP-PMPC closeout metrics.
- It does not include jump smoke.
- It is driven by adaptive PD/VMC panorama logic, not the final GP-PMPC closeout metrics.
- Labels are presentation/demo oriented rather than thesis-style capability annotations.
- The terrain zones are not explicitly separated into six completed/future capability regions.

## Recommendation

Keep the existing panorama scripts unchanged as a dynamic demo layer. Add a new `render_capability_map_v2.py` script that generates a static, reproducible, thesis-style map from code and closeout metrics.
