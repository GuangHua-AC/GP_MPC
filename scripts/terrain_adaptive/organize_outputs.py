from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from _common import ADAPTIVE_DIR, ensure_adaptive_dirs


EXPECTED_RESULTS = {
    "adaptive_pd": ADAPTIVE_DIR / "pd" / "terrain_adaptive_left_obstacle_v0p15_adaptive_pd.npz",
    "adaptive_nn_mpc": ADAPTIVE_DIR / "mpc" / "terrain_adaptive_left_obstacle_v0p15_adaptive_nn_mpc_torch.npz",
    "adaptive_gp_mpc": ADAPTIVE_DIR / "mpc" / "terrain_adaptive_left_obstacle_v0p15_adaptive_gp_mpc.npz",
}

EXPECTED_METRICS = [ADAPTIVE_DIR / "metrics" / "terrain_adaptive_summary.csv"]

EXPECTED_VIDEOS = [
    ADAPTIVE_DIR / "videos" / "01_terrain_adaptive_pd.mp4",
    ADAPTIVE_DIR / "videos" / "01_terrain_adaptive_pd.gif",
    ADAPTIVE_DIR / "videos" / "02_terrain_adaptive_nn_mpc.mp4",
    ADAPTIVE_DIR / "videos" / "02_terrain_adaptive_nn_mpc.gif",
    ADAPTIVE_DIR / "videos" / "03_terrain_adaptive_gp_mpc.mp4",
    ADAPTIVE_DIR / "videos" / "03_terrain_adaptive_gp_mpc.gif",
]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    ensure_adaptive_dirs()
    final = ADAPTIVE_DIR / "final"
    copied = 0

    for kind, src in EXPECTED_RESULTS.items():
        if copy_if_exists(src, final / "results" / kind / src.name):
            copied += 1

    for src in EXPECTED_METRICS:
        if copy_if_exists(src, final / "metrics" / src.name):
            copied += 1

    for src in EXPECTED_VIDEOS:
        if copy_if_exists(src, final / "videos" / src.name):
            copied += 1

    readme = ADAPTIVE_DIR / "README.md"
    lines = [
        "# Terrain Adaptive Outputs",
        "",
        "This folder contains blind/unknown-terrain adaptive runs.",
        "",
        "- `pd/`: online adaptive PD/VMC result files.",
        "- `mpc/`: adaptive NN-MPC and GP-MPC result files.",
        "- `metrics/terrain_adaptive_summary.csv`: comparison table.",
        "- `videos/`: rendered videos.",
        "- `final/`: curated copy of the main result files.",
        "",
        "The controller does not read terrain_diff before acting. It adapts leg_diff from roll and roll_dot feedback after terrain contact.",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"copied={copied}")
    print(f"manifest={readme.resolve()}")


if __name__ == "__main__":
    main()
