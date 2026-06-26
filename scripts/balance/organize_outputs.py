from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from wheel_legged.utils.paths import ROOT, task_output_dir


BALANCE_DIR = task_output_dir("balance")


FINAL_RESULTS = {
    "pd": [
        BALANCE_DIR / "pd" / "balance_external_push_30N_120ms_T1p2_Tp1p5_pd.npz",
    ],
    "nn_mpc": [
        BALANCE_DIR / "mpc" / "balance_external_push_30N_120ms_T1p2_Tp1p5_nn_mpc_torch.npz",
    ],
    "gp_mpc": [
        BALANCE_DIR / "mpc" / "balance_external_push_30N_120ms_T1p2_Tp1p5_gp_mpc.npz",
    ],
}

FINAL_VIDEOS = [
    BALANCE_DIR / "videos" / "01_balance_dynamics_pd.mp4",
    BALANCE_DIR / "videos" / "01_balance_dynamics_pd.gif",
    BALANCE_DIR / "videos" / "02_balance_nn_mpc.mp4",
    BALANCE_DIR / "videos" / "02_balance_nn_mpc.gif",
    BALANCE_DIR / "videos" / "03_balance_gp_mpc.mp4",
    BALANCE_DIR / "videos" / "03_balance_gp_mpc.gif",
]

FINAL_METRICS = [
    BALANCE_DIR / "metrics" / "balance_external_push_summary.csv",
]


def copy_if_exists(src: Path, dst_dir: Path) -> Path | None:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def write_manifest(paths: list[Path]) -> Path:
    manifest = BALANCE_DIR / "README.md"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rel_paths = [p.relative_to(ROOT).as_posix() for p in paths]
    lines = [
        "# Balance outputs",
        "",
        "This folder contains the cleaned final balance artifacts.",
        "",
        "Canonical training files stay in their original locations so scripts keep working:",
        "",
        "- data/balance_transitions.npz",
        "- outputs/balance/models/balance_nn_torch.pt",
        "- outputs/balance/models/balance_gp.joblib",
        "",
        "Final copied artifacts:",
        "",
    ]
    lines += [f"- {p}" for p in rel_paths]
    lines.append("")
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    copied: list[Path] = []

    for controller, paths in FINAL_RESULTS.items():
        for src in paths:
            dst = copy_if_exists(src, BALANCE_DIR / "final" / "results" / controller)
            if dst:
                copied.append(dst)

    for src in FINAL_VIDEOS:
        dst = copy_if_exists(src, BALANCE_DIR / "final" / "videos")
        if dst:
            copied.append(dst)

    for src in FINAL_METRICS:
        dst = copy_if_exists(src, BALANCE_DIR / "final" / "metrics")
        if dst:
            copied.append(dst)

    manifest = write_manifest(copied)
    print(f"copied={len(copied)}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
