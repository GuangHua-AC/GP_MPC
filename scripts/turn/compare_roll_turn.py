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

from _common import angle_error, compute_metrics


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

    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    yaw_refs = np.asarray(data["yaw_refs"], dtype=float) if "yaw_refs" in data.files else np.zeros(len(states))
    roll_refs = np.asarray(data["roll_refs"], dtype=float) if "roll_refs" in data.files else np.zeros(len(states))
    final_reason = str(scalar(data, "final_reason", "unknown"))
    controller = str(scalar(data, "controller", path.stem))
    metrics = compute_metrics(states, actions, yaw_refs, roll_refs, final_reason)

    n = min(len(states), len(yaw_refs), len(roll_refs))
    yaw_rmse = float(np.sqrt(np.mean(angle_error(states[:n, 6], yaw_refs[:n]) ** 2))) if n else 0.0
    roll_rmse = float(np.sqrt(np.mean((states[:n, 8] - roll_refs[:n]) ** 2))) if n else 0.0

    return {
        "controller": controller,
        "target_deg": float(scalar(data, "target_deg", 0.0)),
        "v_ref": float(scalar(data, "v_ref", 0.0)),
        "roll0_deg": float(scalar(data, "roll0_deg", 0.0)),
        "steps": int(metrics["steps"]),
        "final_reason": str(metrics["final_reason"]),
        "success": bool(metrics["success"]),
        "yaw_rmse_deg": float(np.rad2deg(yaw_rmse)),
        "roll_rmse_deg": float(np.rad2deg(roll_rmse)),
        "max_abs_yaw_error_deg": float(np.rad2deg(metrics["max_abs_yaw_error_rad"])),
        "max_abs_roll_error_deg": float(np.rad2deg(metrics["max_abs_roll_error_rad"])),
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
    turn_dir = task_output_dir("balance_turn_roll")
    paths = sorted((turn_dir / "pd").glob("turn_roll_*.npz"))
    paths += sorted((turn_dir / "mpc").glob("turn_roll_*.npz"))
    paths += sorted((turn_dir / "final").rglob("turn_roll_*.npz"))

    deduped: dict[str, Path] = {}
    final_root = turn_dir / "final"
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
    rows.sort(key=lambda r: (float(r["target_deg"]), str(r["controller"]), str(r["file"])))

    out = Path(args.out) if args.out else turn_dir / "metrics" / "turn_roll_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No turn-roll result files found.")
        return

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={out}")
    print("controller, target, success, reason, yaw_rmse, roll_rmse, max_roll")
    for row in rows:
        print(
            f"{row['controller']}, target={row['target_deg']:g}deg, "
            f"success={row['success']}, reason={row['final_reason']}, "
            f"yaw_rmse={row['yaw_rmse_deg']:.3f}deg, "
            f"roll_rmse={row['roll_rmse_deg']:.3f}deg, "
            f"max_roll={row['max_abs_roll_error_deg']:.3f}deg"
        )


if __name__ == "__main__":
    main()
