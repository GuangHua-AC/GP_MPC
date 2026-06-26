from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from wheel_legged.utils.paths import ensure_dirs, task_output_dir

from _common import compute_metrics


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


def read_result(path: Path) -> dict[str, str | float | int | bool] | None:
    data = np.load(path, allow_pickle=True)
    if "states" not in data.files or "actions" not in data.files:
        return None
    if "terrain_diffs" not in data.files or "support_rolls" not in data.files:
        return None
    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    terrain_diffs = np.asarray(data["terrain_diffs"], dtype=float).reshape(-1)
    support_rolls = np.asarray(data["support_rolls"], dtype=float).reshape(-1)
    final_reason = str(scalar(data, "final_reason", "unknown"))
    controller = str(scalar(data, "controller", path.stem))
    metrics = compute_metrics(states, actions, terrain_diffs, support_rolls, final_reason)
    return {
        "controller": controller,
        "terrain_mode": str(scalar(data, "terrain_mode", "unknown")),
        "v_ref": float(scalar(data, "v_ref", 0.0)),
        "steps": int(metrics["steps"]),
        "final_reason": str(metrics["final_reason"]),
        "success": bool(metrics["success"]),
        "max_abs_roll_deg": float(np.rad2deg(metrics["max_abs_roll_rad"])),
        "max_abs_support_roll_deg": float(np.rad2deg(metrics["max_abs_support_roll_rad"])),
        "max_abs_terrain_diff_m": float(metrics["max_abs_terrain_diff_m"]),
        "max_abs_leg_diff_m": float(metrics["max_abs_leg_diff_m"]),
        "max_abs_theta_rad": float(metrics["max_abs_theta_rad"]),
        "max_abs_phi_rad": float(metrics["max_abs_phi_rad"]),
        "max_abs_x_m": float(metrics["max_abs_x_m"]),
        "action_energy": float(metrics["action_energy"]),
        "file": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_dirs()
    terrain_dir = task_output_dir("terrain")
    paths = sorted((terrain_dir / "pd").glob("terrain_*.npz"))
    paths += sorted((terrain_dir / "mpc").glob("terrain_*.npz"))
    paths += sorted((terrain_dir / "final").rglob("terrain_*.npz"))
    final_root = terrain_dir / "final"
    deduped: dict[str, Path] = {}
    for path in paths:
        old = deduped.get(path.name)
        if old is None:
            deduped[path.name] = path
            continue
        old_is_final = final_root in old.parents
        path_is_final = final_root in path.parents
        path_is_newer = path.stat().st_mtime > old.stat().st_mtime
        path_is_same_time_working_copy = path.stat().st_mtime == old.stat().st_mtime and old_is_final and not path_is_final
        if path_is_newer or path_is_same_time_working_copy:
            deduped[path.name] = path

    rows = [row for path in sorted(deduped.values()) if (row := read_result(path)) is not None]
    rows.sort(key=lambda r: (str(r["terrain_mode"]), str(r["controller"]), str(r["file"])))

    out = Path(args.out) if args.out else terrain_dir / "metrics" / "terrain_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No terrain result files found.")
        return
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={out}")
    print("controller, terrain, success, reason, max_roll, max_support_roll, max_leg_diff")
    for row in rows:
        print(
            f"{row['controller']}, terrain={row['terrain_mode']}, "
            f"success={row['success']}, reason={row['final_reason']}, "
            f"max_roll={row['max_abs_roll_deg']:.3f}deg, "
            f"max_support_roll={row['max_abs_support_roll_deg']:.3f}deg, "
            f"max_leg_diff={row['max_abs_leg_diff_m']:.4f}m"
        )


if __name__ == "__main__":
    main()
