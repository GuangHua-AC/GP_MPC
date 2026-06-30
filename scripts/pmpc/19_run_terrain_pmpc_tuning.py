from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pmpc" / "17_run_python_terrain_pmpc.py"
OUT_DIR = ROOT / "outputs" / "terrain" / "pmpc" / "tuning"


def tag(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:g}".replace(".", "p").replace("-", "m")


def output_path(case: dict) -> Path:
    return OUT_DIR / (
        f"{case['name']}_terrain_{case['terrain_mode']}_H{case['horizon']}_C{case['candidates']}_"
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
        "--terrain-mode",
        case["terrain_mode"],
        "--obstacle-height",
        str(case["obstacle_height"]),
        "--obstacle-start",
        str(case["obstacle_start"]),
        "--obstacle-length",
        str(case["obstacle_length"]),
        "--T-limit",
        str(case["T_limit"]),
        "--Tp-limit",
        str(case["Tp_limit"]),
        "--x-limit",
        str(case["x_limit"]),
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
        "terrain_mode": "left_obstacle",
        "obstacle_height": 0.04,
        "obstacle_start": 1.0,
        "obstacle_length": 0.5,
        "T_limit": 1.2,
        "Tp_limit": 1.5,
        "x_limit": 2.0,
        "uncertainty_weight": 5,
        "terminal_weight": 0,
        "k_sigma": 2,
        "seed": 0,
    }
    cases = [
        {**base, "name": "A_baseline_v0", "horizon": 4, "candidates": 16, "chance_weight": 20, "guide_weight": 20, "noise_scale": 0.25, "random_fraction": 0.15},
        {**base, "name": "B_terrain_low_noise", "horizon": 4, "candidates": 16, "chance_weight": 20, "guide_weight": 20, "noise_scale": 0.03, "random_fraction": 0.0},
        {**base, "name": "C_guide_stronger", "horizon": 4, "candidates": 16, "chance_weight": 20, "guide_weight": 50, "noise_scale": 0.03, "random_fraction": 0.0},
        {**base, "name": "D_chance_stronger", "horizon": 4, "candidates": 16, "chance_weight": 50, "guide_weight": 20, "noise_scale": 0.03, "random_fraction": 0.0},
        {**base, "name": "E_chance_guide_stronger", "horizon": 4, "candidates": 16, "chance_weight": 50, "guide_weight": 50, "noise_scale": 0.03, "random_fraction": 0.0},
        {**base, "name": "F_slightly_larger_search", "horizon": 6, "candidates": 32, "chance_weight": 20, "guide_weight": 50, "noise_scale": 0.03, "random_fraction": 0.0},
    ]
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
