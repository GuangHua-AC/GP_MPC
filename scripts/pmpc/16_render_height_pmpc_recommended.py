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


STEP_FILE = OUTPUT_DIR / "height" / "pmpc" / "tracking" / "B_step_height_pmpc_step_Uw5_Cw20_Gw20_Tw0_K2_seed0.npz"
SINE_FILE = OUTPUT_DIR / "height" / "pmpc" / "tracking" / "C_sine_height_pmpc_sine_Uw5_Cw20_Gw20_Tw0_K2_seed0.npz"


def find_recommended_npz(prefer: str) -> Path:
    ordered = [STEP_FILE, SINE_FILE] if prefer == "step" else [SINE_FILE, STEP_FILE]
    for path in ordered:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Recommended height GP-PMPC tracking result not found. Run:\n"
        "python scripts/pmpc/14_run_height_pmpc_tracking_set.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefer", choices=["step", "sine"], default="step")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--out", default=str(OUTPUT_DIR / "height" / "videos" / "04_height_gp_pmpc_step_recommended.mp4"))
    parser.add_argument("--gif", default=str(OUTPUT_DIR / "height" / "videos" / "04_height_gp_pmpc_step_recommended.gif"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    npz = Path(args.npz) if args.npz else find_recommended_npz(args.prefer)
    cmd = [
        sys.executable,
        "scripts/height/render_height_result.py",
        "--npz",
        str(npz),
        "--out",
        args.out,
        "--gif",
        args.gif,
        "--stride",
        str(args.stride),
        "--speed",
        str(args.speed),
    ]
    print("title=Height GP-PMPC + safety-guided action regularization, L0 step tracking")
    print("RUN", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print("render_failed")
        print("If MP4/GIF encoding is unavailable, rerun with only one output or install ffmpeg/Pillow.")
        raise SystemExit(exc.returncode) from exc

    print(f"recommended_npz={npz}")
    print(f"saved_mp4={args.out}")
    print(f"saved_gif={args.gif}")


if __name__ == "__main__":
    main()
