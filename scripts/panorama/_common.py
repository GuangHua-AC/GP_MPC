from __future__ import annotations

from pathlib import Path

from wheel_legged.utils.paths import OUTPUT_DIR


PANORAMA_DIR = OUTPUT_DIR / "panorama"


def ensure_panorama_dirs() -> None:
    for name in ["results", "metrics", "videos", "final"]:
        (PANORAMA_DIR / name).mkdir(parents=True, exist_ok=True)
