from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from _common import PANORAMA_DIR, ensure_panorama_dirs


EXPECTED = [
    PANORAMA_DIR / "results" / "panorama_showcase_adaptive_pd.npz",
    PANORAMA_DIR / "metrics" / "panorama_summary.csv",
    PANORAMA_DIR / "videos" / "panorama_showcase.mp4",
    PANORAMA_DIR / "videos" / "panorama_showcase.gif",
]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    ensure_panorama_dirs()
    copied = 0
    for src in EXPECTED:
        if src.suffix == ".npz":
            dst = PANORAMA_DIR / "final" / "results" / src.name
        elif src.suffix == ".csv":
            dst = PANORAMA_DIR / "final" / "metrics" / src.name
        else:
            dst = PANORAMA_DIR / "final" / "videos" / src.name
        if copy_if_exists(src, dst):
            copied += 1
    readme = PANORAMA_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Panorama Outputs",
                "",
                "Full-map showcase combining balance, roll turn, height change, and adaptive terrain.",
                "",
                "- `results/`: panorama trajectory npz.",
                "- `metrics/`: summary csv.",
                "- `videos/`: rendered overview video.",
                "- `final/`: curated copies.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"copied={copied}")
    print(f"manifest={readme.resolve()}")


if __name__ == "__main__":
    main()
