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


RECOMMENDED_GLOB = "*Uw5_Cw20_Gw20_K2_Tw0_seed0_pushStart1*_gp_pmpc.npz"
FALLBACK_ABLATION = "G_chance_guide_Uw5_Cw20_Gw20_Tw0_seed0.npz"


def find_recommended_npz() -> Path:
    pmpc_dir = OUTPUT_DIR / "balance" / "pmpc"
    candidates = sorted(path for path in pmpc_dir.glob(RECOMMENDED_GLOB) if path.is_file())
    if candidates:
        return candidates[-1]

    fallback = pmpc_dir / "ablation" / FALLBACK_ABLATION
    if fallback.exists():
        print(f"recommended_top_level_missing_using_ablation={fallback}")
        return fallback

    raise FileNotFoundError(
        "Recommended GP-PMPC result not found. Run:\n"
        "python scripts/pmpc/01_run_python_balance_pmpc.py --horizon 8 --candidates 96 "
        "--uncertainty-weight 5 --chance-weight 20 --guide-weight 20 --terminal-weight 0 "
        "--push-start 1.0 --seed 0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=None)
    parser.add_argument("--out", default=str(OUTPUT_DIR / "balance" / "videos" / "04_balance_gp_pmpc_recommended.mp4"))
    parser.add_argument("--gif", default=str(OUTPUT_DIR / "balance" / "videos" / "04_balance_gp_pmpc_recommended.gif"))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--title", default="GP-PMPC + safety-guided action regularization, 30N push")
    args = parser.parse_args()

    npz = Path(args.npz) if args.npz else find_recommended_npz()
    cmd = [
        sys.executable,
        "scripts/balance/render_result.py",
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
        "--title",
        args.title,
    ]
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
