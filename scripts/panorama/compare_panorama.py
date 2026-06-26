from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from _common import PANORAMA_DIR, ensure_panorama_dirs


def scalar(data, key: str, default=None):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    ensure_panorama_dirs()
    rows = []
    for path in sorted((PANORAMA_DIR / "results").glob("panorama_*.npz")):
        data = np.load(path, allow_pickle=False)
        states = np.asarray(data["states"], dtype=float)
        terrain_diffs = np.asarray(data["terrain_diffs"], dtype=float)
        support_rolls = np.asarray(data["support_rolls"], dtype=float)
        phase_names = np.asarray(data["phase_names"]).reshape(-1)
        rows.append(
            {
                "controller": str(scalar(data, "controller", path.stem)),
                "steps": int(len(states)),
                "final_reason": str(scalar(data, "final_reason", "unknown")),
                "success": bool(scalar(data, "success", False)),
                "phases": "|".join(str(x) for x in phase_names),
                "max_abs_theta_rad": float(np.max(np.abs(states[:, 0]))),
                "max_abs_phi_rad": float(np.max(np.abs(states[:, 4]))),
                "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(states[:, 8])))),
                "max_abs_support_roll_deg": float(np.rad2deg(np.max(np.abs(support_rolls)))),
                "max_abs_terrain_diff_m": float(np.max(np.abs(terrain_diffs))),
                "max_abs_leg_diff_m": float(np.max(np.abs(states[:, 12]))),
                "max_abs_x_m": float(np.max(np.abs(states[:, 2]))),
                "file": str(path.resolve()),
            }
        )

    out = PANORAMA_DIR / "metrics" / "panorama_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No panorama result files found.")
        return
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={out.resolve()}")
    for row in rows:
        print(
            f"{row['controller']}, success={row['success']}, reason={row['final_reason']}, "
            f"max_roll={row['max_abs_roll_deg']:.3f}deg, max_support_roll={row['max_abs_support_roll_deg']:.3f}deg, "
            f"max_leg_diff={row['max_abs_leg_diff_m']:.4f}m"
        )


if __name__ == "__main__":
    main()
