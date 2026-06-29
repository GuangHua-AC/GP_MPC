from __future__ import annotations

import argparse
import shutil
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
ABLATION_DIR = OUTPUT_DIR / "balance" / "pmpc" / "ablation"


def _run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _copy_original_result(method: str, uncertainty_weight: float) -> Path:
    src = OUTPUT_DIR / "balance" / "mpc" / "balance_external_push_30N_120ms_T1p2_Tp1p5_gp_mpc.npz"
    out = ABLATION_DIR / f"{method}.npz"
    if not src.exists():
        raise FileNotFoundError(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    print(f"copied_original_result={out} uncertainty_weight={uncertainty_weight}")
    return out


def run_original_gp_mpc(uncertainty_weight: float, method: str, args) -> Path:
    _run(
        [
            PYTHON,
            "scripts/balance/test_external_push_gp_mpc.py",
            "--push-force",
            str(args.push_force),
            "--push-duration",
            str(args.push_duration),
            "--steps",
            str(args.steps),
            "--horizon",
            str(args.horizon),
            "--candidates",
            str(args.candidates),
            "--uncertainty-weight",
            str(uncertainty_weight),
            "--Tp-limit",
            str(args.Tp_limit),
            "--x-limit",
            str(args.x_limit),
        ]
    )
    return _copy_original_result(method, uncertainty_weight)


def run_pmpc(uncertainty_weight: float, chance_weight: float, guide_weight: float, method: str, args, seed: int | None = None) -> Path:
    out = ABLATION_DIR / f"{method}.npz"
    run_seed = args.seed if seed is None else seed
    _run(
        [
            PYTHON,
            "scripts/pmpc/01_run_python_balance_pmpc.py",
            "--steps",
            str(args.steps),
            "--horizon",
            str(args.horizon),
            "--candidates",
            str(args.candidates),
            "--uncertainty-weight",
            str(uncertainty_weight),
            "--chance-weight",
            str(chance_weight),
            "--guide-weight",
            str(guide_weight),
            "--terminal-weight",
            str(args.terminal_weight),
            "--k-sigma",
            str(args.k_sigma),
            "--push-force",
            str(args.push_force),
            "--push-duration",
            str(args.push_duration),
            "--push-start",
            str(args.push_start),
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


def restore_archived_gp_mpc() -> None:
    src = OUTPUT_DIR / "balance" / "final" / "results" / "gp_mpc" / "balance_external_push_30N_120ms_T1p2_Tp1p5_gp_mpc.npz"
    dst = OUTPUT_DIR / "balance" / "mpc" / "balance_external_push_30N_120ms_T1p2_Tp1p5_gp_mpc.npz"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"restored_archived_gp_mpc={dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--push-force", type=float, default=30.0)
    parser.add_argument("--push-duration", type=float, default=0.12)
    parser.add_argument("--push-start", type=float, default=1.0)
    parser.add_argument("--T-limit", type=float, default=1.2)
    parser.add_argument("--Tp-limit", type=float, default=1.5)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-sigma", type=float, default=2.0)
    parser.add_argument("--terminal-weight", type=float, default=0.0)
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for path in ABLATION_DIR.glob("*.npz"):
            path.unlink()

    if not args.skip_original:
        run_original_gp_mpc(0.0, "A_original_gp_mpc_Uw0", args)
        run_original_gp_mpc(5.0, "B_original_gp_mpc_Uw5", args)
        restore_archived_gp_mpc()

    run_pmpc(0.0, 0.0, 0.0, "C_pmpc_Uw0_Cw0_Gw0_Tw0_seed0", args, seed=0)
    run_pmpc(5.0, 0.0, 0.0, "D_pmpc_Uw5_Cw0_Gw0_Tw0_seed0", args, seed=0)
    run_pmpc(5.0, 20.0, 0.0, "E_chance_only_Uw5_Cw20_Gw0_Tw0_seed0", args, seed=0)
    run_pmpc(5.0, 0.0, 20.0, "F_guide_only_Uw5_Cw0_Gw20_Tw0_seed0", args, seed=0)
    run_pmpc(5.0, 20.0, 20.0, "G_chance_guide_Uw5_Cw20_Gw20_Tw0_seed0", args, seed=0)
    run_pmpc(5.0, 50.0, 20.0, "H_chance_guide_Uw5_Cw50_Gw20_Tw0_seed0", args, seed=0)
    for seed in [1, 2]:
        run_pmpc(5.0, 20.0, 20.0, f"G_chance_guide_Uw5_Cw20_Gw20_Tw0_seed{seed}", args, seed=seed)

    print(f"ablation_dir={ABLATION_DIR}")


if __name__ == "__main__":
    main()
