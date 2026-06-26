from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from wheel_legged.utils.paths import ROOT, task_output_dir


TERRAIN_DIR = task_output_dir("terrain")

FINAL_RESULTS = {
    "pd": TERRAIN_DIR / "pd" / "terrain_left_obstacle_v0p15_pd.npz",
    "nn_mpc": TERRAIN_DIR / "mpc" / "terrain_left_obstacle_v0p15_nn_mpc_torch.npz",
    "gp_mpc": TERRAIN_DIR / "mpc" / "terrain_left_obstacle_v0p15_gp_mpc.npz",
}

FINAL_METRICS = [TERRAIN_DIR / "metrics" / "terrain_summary.csv"]
FINAL_VIDEOS = [
    TERRAIN_DIR / "videos" / "01_terrain_pd.mp4",
    TERRAIN_DIR / "videos" / "01_terrain_pd.gif",
    TERRAIN_DIR / "videos" / "02_terrain_nn_mpc.mp4",
    TERRAIN_DIR / "videos" / "02_terrain_nn_mpc.gif",
    TERRAIN_DIR / "videos" / "03_terrain_gp_mpc.mp4",
    TERRAIN_DIR / "videos" / "03_terrain_gp_mpc.gif",
]


def copy_if_exists(src: Path, dst_dir: Path) -> Path | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def write_manifest(paths: list[Path]) -> Path:
    manifest = TERRAIN_DIR / "README.md"
    rel_paths = [p.relative_to(ROOT).as_posix() for p in paths]
    lines = ["# Terrain outputs", "", "Final copied artifacts:", ""]
    lines += [f"- {p}" for p in rel_paths]
    lines.append("")
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    copied: list[Path] = []
    for name, src in FINAL_RESULTS.items():
        dst = copy_if_exists(src, TERRAIN_DIR / "final" / "results" / name)
        if dst:
            copied.append(dst)
    for src in FINAL_METRICS:
        dst = copy_if_exists(src, TERRAIN_DIR / "final" / "metrics")
        if dst:
            copied.append(dst)
    for src in FINAL_VIDEOS:
        dst = copy_if_exists(src, TERRAIN_DIR / "final" / "videos")
        if dst:
            copied.append(dst)
    manifest = write_manifest(copied)
    print(f"copied={len(copied)}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
