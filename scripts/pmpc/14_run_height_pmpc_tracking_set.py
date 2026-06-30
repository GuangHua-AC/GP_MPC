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
TRACKING_DIR = OUTPUT_DIR / "height" / "pmpc" / "tracking"


def _run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_case(label: str, mode: str, args) -> Path:
    out = TRACKING_DIR / f"{label}_height_pmpc_{mode}_Uw5_Cw20_Gw20_Tw0_K2_seed{args.seed}.npz"
    cmd = [
        PYTHON,
        "scripts/pmpc/12_run_python_height_pmpc.py",
        "--mode",
        mode,
        "--steps",
        str(args.steps),
        "--horizon",
        str(args.horizon),
        "--candidates",
        str(args.candidates),
        "--uncertainty-weight",
        str(args.uncertainty_weight),
        "--chance-weight",
        str(args.chance_weight),
        "--guide-weight",
        str(args.guide_weight),
        "--terminal-weight",
        str(args.terminal_weight),
        "--k-sigma",
        str(args.k_sigma),
        "--noise-scale",
        str(args.noise_scale),
        "--random-fraction",
        str(args.random_fraction),
        "--T-limit",
        str(args.T_limit),
        "--Tp-limit",
        str(args.Tp_limit),
        "--x-limit",
        str(args.x_limit),
        "--seed",
        str(args.seed),
        "--out",
        str(out),
    ]
    if mode == "fixed":
        cmd += ["--L0-ref", "0.34", "--low", "0.30", "--high", "0.34"]
    elif mode == "step":
        cmd += ["--low", "0.30", "--high", "0.34", "--switch-time", str(args.switch_time)]
    elif mode == "sine":
        cmd += ["--low", "0.30", "--high", "0.34", "--period", str(args.period)]
    _run(cmd)
    return out


def summarize(path: Path) -> str:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    final_reason = str(data["final_reason"].reshape(-1)[0])
    return (
        f"{path.stem},{len(data['states'])},{final_reason},"
        f"{float(data['max_abs_L0_error_after_1s'].reshape(-1)[0]):.5f},"
        f"{float(data['rmse_L0_error_after_1s'].reshape(-1)[0]):.5f},"
        f"{float(data['final_L0_error'].reshape(-1)[0]):.5f},"
        f"{float(data['max_abs_theta'].reshape(-1)[0]):.4f},"
        f"{float(data['max_abs_phi'].reshape(-1)[0]):.4f},"
        f"{float(data['max_abs_x'].reshape(-1)[0]):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--uncertainty-weight", type=float, default=5.0)
    parser.add_argument("--chance-weight", type=float, default=20.0)
    parser.add_argument("--guide-weight", type=float, default=20.0)
    parser.add_argument("--terminal-weight", type=float, default=0.0)
    parser.add_argument("--k-sigma", type=float, default=2.0)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--random-fraction", type=float, default=0.0)
    parser.add_argument("--T-limit", type=float, default=1.2)
    parser.add_argument("--Tp-limit", type=float, default=1.5)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--switch-time", type=float, default=2.0)
    parser.add_argument("--period", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for path in TRACKING_DIR.glob("*.npz"):
            path.unlink()

    outputs = [
        run_case("A_fixed", "fixed", args),
        run_case("B_step", "step", args),
        run_case("C_sine", "sine", args),
    ]
    print("method,steps,final_reason,max_abs_L0_error_after_1s,rmse_L0_error_after_1s,final_L0_error,max_abs_theta,max_abs_phi,max_abs_x")
    for path in outputs:
        print(summarize(path))
    print(f"tracking_dir={TRACKING_DIR}")


if __name__ == "__main__":
    main()
