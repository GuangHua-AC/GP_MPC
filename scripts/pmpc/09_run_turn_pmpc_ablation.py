from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from wheel_legged.utils.paths import OUTPUT_DIR


PYTHON = sys.executable
ABLATION_DIR = OUTPUT_DIR / "turn" / "pmpc" / "ablation"


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_case(label: str, uw: float, cw: float, gw: float, args, seed: int | None = None) -> Path:
    run_seed = args.seed if seed is None else seed
    out = (
        ABLATION_DIR
        / (
            f"{label}_target{_tag(args.target_deg)}deg_v{_tag(args.v_ref)}_"
            f"Uw{_tag(uw)}_Cw{_tag(cw)}_Gw{_tag(gw)}_Tw{_tag(args.terminal_weight)}_"
            f"K{_tag(args.k_sigma)}_seed{run_seed}.npz"
        )
    )
    _run(
        [
            PYTHON,
            "scripts/pmpc/07_run_python_turn_pmpc.py",
            "--steps",
            str(args.steps),
            "--target-deg",
            str(args.target_deg),
            "--v-ref",
            str(args.v_ref),
            "--horizon",
            str(args.horizon),
            "--candidates",
            str(args.candidates),
            "--uncertainty-weight",
            str(uw),
            "--chance-weight",
            str(cw),
            "--guide-weight",
            str(gw),
            "--terminal-weight",
            str(args.terminal_weight),
            "--k-sigma",
            str(args.k_sigma),
            "--T-limit",
            str(args.T_limit),
            "--Tp-limit",
            str(args.Tp_limit),
            "--x-limit",
            str(args.x_limit),
            "--seed",
            str(run_seed),
            "--out",
            str(out),
        ]
    )
    return out


def summarize(path: Path) -> str:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    states = data["states"]
    final_reason = str(data["final_reason"].reshape(-1)[0]) if "final_reason" in data.files else "unknown"
    final_yaw_error = float(data["final_yaw_error_deg"].reshape(-1)[0]) if "final_yaw_error_deg" in data.files else 0.0
    max_roll = float(data["max_abs_roll"].reshape(-1)[0]) if "max_abs_roll" in data.files else float(np.max(np.abs(states[:, 8])))
    max_theta = float(data["max_abs_theta"].reshape(-1)[0]) if "max_abs_theta" in data.files else float(np.max(np.abs(states[:, 0])))
    max_phi = float(data["max_abs_phi"].reshape(-1)[0]) if "max_abs_phi" in data.files else float(np.max(np.abs(states[:, 4])))
    max_x = float(data["max_abs_x"].reshape(-1)[0]) if "max_abs_x" in data.files else float(np.max(np.abs(states[:, 2])))
    return (
        f"{path.stem},{len(states)},{final_reason},{final_yaw_error:.3f},"
        f"{max_roll:.4f},{max_theta:.4f},{max_phi:.4f},{max_x:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--target-deg", type=float, default=30.0)
    parser.add_argument("--v-ref", type=float, default=0.15)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--T-limit", type=float, default=1.2)
    parser.add_argument("--Tp-limit", type=float, default=1.5)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-sigma", type=float, default=2.0)
    parser.add_argument("--terminal-weight", type=float, default=0.0)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for path in ABLATION_DIR.glob("*.npz"):
            path.unlink()

    outputs = [
        run_case("A_recommended", 5.0, 20.0, 20.0, args, seed=0),
        run_case("B_chance_only", 5.0, 20.0, 0.0, args, seed=0),
        run_case("C_guide_only", 5.0, 0.0, 20.0, args, seed=0),
        run_case("D_uncertainty_only", 5.0, 0.0, 0.0, args, seed=0),
        run_case("E_mean_only", 0.0, 0.0, 0.0, args, seed=0),
    ]
    for seed in [1, 2]:
        outputs.append(run_case("A_recommended", 5.0, 20.0, 20.0, args, seed=seed))

    print("method,steps,final_reason,final_yaw_error_deg,max_abs_roll,max_abs_theta,max_abs_phi,max_abs_x")
    for path in outputs:
        print(summarize(path))
    print(f"ablation_dir={ABLATION_DIR}")


if __name__ == "__main__":
    main()
