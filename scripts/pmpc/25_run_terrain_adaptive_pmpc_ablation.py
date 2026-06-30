from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pmpc" / "23_run_python_terrain_adaptive_pmpc.py"
OUT_DIR = ROOT / "outputs" / "terrain_adaptive" / "pmpc" / "ablation"


def tag(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:g}".replace(".", "p").replace("-", "m")


def output_path(case: dict) -> Path:
    return OUT_DIR / (
        f"{case['name']}_terrain_adaptive_{case['terrain_mode']}_H{case['horizon']}_C{case['candidates']}_"
        f"Ag{tag(case['adaptive_gain'])}_Alim{tag(case['adaptive_limit'])}_"
        f"Uw{tag(case['uncertainty_weight'])}_Cw{tag(case['chance_weight'])}_Gw{tag(case['guide_weight'])}_"
        f"N{tag(case['noise_scale'])}_Rf{tag(case['random_fraction'])}_seed{case['seed']}.npz"
    )


def run_case(case: dict) -> None:
    out = output_path(case)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--steps",
        str(case["steps"]),
        "--horizon",
        str(case["horizon"]),
        "--candidates",
        str(case["candidates"]),
        "--uncertainty-weight",
        str(case["uncertainty_weight"]),
        "--chance-weight",
        str(case["chance_weight"]),
        "--guide-weight",
        str(case["guide_weight"]),
        "--terminal-weight",
        str(case["terminal_weight"]),
        "--k-sigma",
        str(case["k_sigma"]),
        "--noise-scale",
        str(case["noise_scale"]),
        "--random-fraction",
        str(case["random_fraction"]),
        "--terrain-mode",
        case["terrain_mode"],
        "--obstacle-height",
        str(case["obstacle_height"]),
        "--obstacle-start",
        str(case["obstacle_start"]),
        "--obstacle-length",
        str(case["obstacle_length"]),
        "--adaptive-gain",
        str(case["adaptive_gain"]),
        "--adaptive-limit",
        str(case["adaptive_limit"]),
        "--seed",
        str(case["seed"]),
        "--out",
        str(out),
    ]
    print(f"running={case['name']}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    base = {
        "steps": 1200,
        "horizon": 4,
        "candidates": 16,
        "uncertainty_weight": 5,
        "chance_weight": 200,
        "guide_weight": 50,
        "terminal_weight": 0,
        "k_sigma": 2,
        "noise_scale": 0.03,
        "random_fraction": 0.0,
        "terrain_mode": "left_obstacle",
        "obstacle_height": 0.04,
        "obstacle_start": 1.0,
        "obstacle_length": 0.5,
        "seed": 0,
    }
    cases = [
        {**base, "name": "A_no_adaptive", "adaptive_gain": 0.0, "adaptive_limit": 0.08},
        {**base, "name": "B_adaptive_gain_0p3", "adaptive_gain": 0.3, "adaptive_limit": 0.08},
        {**base, "name": "C_adaptive_gain_0p5", "adaptive_gain": 0.5, "adaptive_limit": 0.08},
        {**base, "name": "D_adaptive_gain_0p8", "adaptive_gain": 0.8, "adaptive_limit": 0.08},
        {**base, "name": "E_adaptive_limit_0p06", "adaptive_gain": 0.5, "adaptive_limit": 0.06},
        {**base, "name": "F_adaptive_limit_0p10", "adaptive_gain": 0.5, "adaptive_limit": 0.10},
    ]
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
