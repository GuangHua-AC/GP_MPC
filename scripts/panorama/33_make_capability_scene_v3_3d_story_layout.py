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
    run([py, "scripts/panorama/run_capability_scene_v3_3d_story.py", "--variant", "layout"])
    run(
        [
            py,
            "scripts/panorama/render_capability_scene_v3_3d_story.py",
            "--input",
            "outputs/panorama/capability_scene/capability_scene_v3_3d_story_layout.npz",
            "--out-mp4",
            "outputs/panorama/videos/capability_scene_v3_3d_story_layout.mp4",
            "--out-gif",
            "outputs/panorama/videos/capability_scene_v3_3d_story_layout.gif",
            "--snapshot",
            "outputs/panorama/figures/capability_scene_v3_3d_story_layout_snapshot.png",
            "--debug-clearance",
        ]
    )


if __name__ == "__main__":
    main()
