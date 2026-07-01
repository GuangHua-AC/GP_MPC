from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    print("running=" + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    py = sys.executable
    run([py, "scripts/panorama/run_capability_scene_v2.py"])
    run([py, "scripts/panorama/render_capability_scene_v2.py"])


if __name__ == "__main__":
    main()
