from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

TASK_OUTPUT_GROUPS = {
    "balance": "balance",
    "balance_turn": "turn",
    "balance_turn_roll": "turn",
    "height": "height",
    "terrain": "terrain",
}


def output_group(task: str) -> str:
    return TASK_OUTPUT_GROUPS.get(task, task)


def task_output_dir(task: str) -> Path:
    return OUTPUT_DIR / output_group(task)


def task_output_subdir(task: str, name: str) -> Path:
    path = task_output_dir(task) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs() -> None:
    base_dirs = [DATA_DIR, OUTPUT_DIR]
    task_dirs = []
    for group in sorted(set(TASK_OUTPUT_GROUPS.values())):
        root = OUTPUT_DIR / group
        task_dirs.extend([root, root / "pd", root / "mpc", root / "models", root / "metrics", root / "videos"])
    for path in [*base_dirs, *task_dirs]:
        path.mkdir(parents=True, exist_ok=True)
