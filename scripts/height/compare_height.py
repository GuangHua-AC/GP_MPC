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

from _common import compute_l0s, compute_metrics


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
    if "L0_refs" not in data.files or "mode" not in data.files or "controller" not in data.files:
        return None

    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    l0s = np.asarray(data["L0s"], dtype=float).reshape(-1) if "L0s" in data.files else compute_l0s(states)
    l0_refs = np.asarray(data["L0_refs"], dtype=float).reshape(-1)
    final_reason = str(scalar(data, "final_reason", "unknown"))
    controller = str(scalar(data, "controller", path.stem))
    metrics = compute_metrics(states, actions, l0s, l0_refs, final_reason)

    return {
        "controller": controller,
        "mode": str(scalar(data, "mode", "unknown")),
        "low": float(scalar(data, "low", 0.0)),
        "high": float(scalar(data, "high", 0.0)),
        "v_ref": float(scalar(data, "v_ref", 0.0)),
        "steps": int(metrics["steps"]),
        "final_reason": str(metrics["final_reason"]),
        "success": bool(metrics["success"]),
        "l0_rmse_m": float(metrics["l0_rmse_m"]),
        "max_abs_l0_error_m": float(metrics["max_abs_l0_error_m"]),
        "final_l0_error_m": float(metrics["final_l0_error_m"]),
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
    height_dir = task_output_dir("height")
    paths = sorted((height_dir / "pd").glob("height_*.npz"))
    paths += sorted((height_dir / "mpc").glob("height_*.npz"))
    paths += sorted((height_dir / "final").rglob("height_*.npz"))

    final_root = height_dir / "final"
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
    rows.sort(key=lambda r: (str(r["mode"]), float(r["v_ref"]), str(r["controller"]), str(r["file"])))

    out = Path(args.out) if args.out else height_dir / "metrics" / "height_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No height result files found.")
        return

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={out}")
    print("controller, mode, v_ref, success, reason, l0_rmse, max_l0_err, final_l0_err")
    for row in rows:
        print(
            f"{row['controller']}, mode={row['mode']}, v_ref={row['v_ref']:.3f}, "
            f"success={row['success']}, reason={row['final_reason']}, "
            f"l0_rmse={row['l0_rmse_m']:.4f}m, "
            f"max_l0_err={row['max_abs_l0_error_m']:.4f}m, "
            f"final_l0_err={row['final_l0_error_m']:.4f}m"
        )


if __name__ == "__main__":
    main()
