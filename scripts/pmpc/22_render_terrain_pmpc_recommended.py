from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TUNING_DIR = ROOT / "outputs" / "terrain" / "pmpc" / "tuning"
VIDEOS_DIR = ROOT / "outputs" / "terrain" / "videos"


def find_recommended_seed0() -> Path | None:
    patterns = [
        "H_chance200_guide50_*seed0.npz",
        "*Cw200_Gw50*N0p03*Rf0*seed0*.npz",
        "*Uw5*Cw200*Gw50*seed0*.npz",
    ]
    for pattern in patterns:
        matches = sorted(TUNING_DIR.glob(pattern))
        if matches:
            return matches[-1]
    return None


def main() -> None:
    npz = find_recommended_seed0()
    if npz is None:
        print(f"missing_recommended_seed0={TUNING_DIR}")
        print(
            "Please run:\n"
            "python scripts/pmpc/17_run_python_terrain_pmpc.py --steps 1200 --horizon 4 --candidates 16 "
            "--uncertainty-weight 5 --chance-weight 200 --guide-weight 50 --terminal-weight 0 "
            "--noise-scale 0.03 --random-fraction 0.0 --seed 0"
        )
        raise SystemExit(2)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out_mp4 = VIDEOS_DIR / "04_terrain_gp_pmpc_recommended.mp4"
    out_gif = VIDEOS_DIR / "04_terrain_gp_pmpc_recommended.gif"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "terrain" / "render_terrain_result.py"),
        "--npz",
        str(npz),
        "--out",
        str(out_mp4),
        "--gif",
        str(out_gif),
        "--stride",
        "10",
        "--speed",
        "1.0",
    ]
    print("title=Known Terrain GP-PMPC + safety-guided action regularization")
    print(f"source={npz}")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"render_failed_exit_code={exc.returncode}")
        print("If ffmpeg or pillow is missing, inspect the traceback above; other closeout artifacts are unaffected.")
        raise
    print(f"saved_mp4={out_mp4}")
    print(f"saved_gif={out_gif}")


if __name__ == "__main__":
    main()
