from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "outputs" / "terrain_adaptive"


def find_recommended_seed0() -> Path | None:
    patterns = [
        "pmpc/terrain_adaptive_pmpc_*Ag0p5_Alim0p08*seed0*.npz",
        "pmpc/ablation/C_adaptive_gain_0p5_*seed0.npz",
        "pmpc/*seed0*.npz",
    ]
    for pattern in patterns:
        matches = sorted(ADAPTIVE_DIR.glob(pattern))
        if matches:
            return matches[-1]
    return None


def main() -> None:
    npz = find_recommended_seed0()
    if npz is None:
        print(f"missing_recommended_seed0={ADAPTIVE_DIR / 'pmpc'}")
        raise SystemExit(2)
    out_mp4 = ADAPTIVE_DIR / "videos" / "04_terrain_adaptive_gp_pmpc_recommended.mp4"
    out_gif = ADAPTIVE_DIR / "videos" / "04_terrain_adaptive_gp_pmpc_recommended.gif"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
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
    print("title=Terrain Adaptive GP-PMPC + safety-guided action regularization")
    print(f"source={npz}")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"render_failed_exit_code={exc.returncode}")
        print("If ffmpeg or pillow is missing, other closeout artifacts are unaffected.")
        raise
    print(f"saved_mp4={out_mp4}")
    print(f"saved_gif={out_gif}")


if __name__ == "__main__":
    main()
