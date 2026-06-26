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

from _common import ADAPTIVE_DIR, compute_metrics, ensure_adaptive_dirs


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


def summarize_file(path: Path) -> dict | None:
    data = np.load(path, allow_pickle=False)
    if "terrain_diffs" not in data.files or "support_rolls" not in data.files:
        return None
    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    terrain_diffs = np.asarray(data["terrain_diffs"], dtype=float).reshape(-1)
    support_rolls = np.asarray(data["support_rolls"], dtype=float).reshape(-1)
    final_reason = str(scalar(data, "final_reason", "unknown"))
    metrics = compute_metrics(states, actions, terrain_diffs, support_rolls, final_reason)
    return {
        "controller": str(scalar(data, "controller", path.stem)),
        "terrain_mode": str(scalar(data, "terrain_mode", "unknown")),
        "terrain_known_to_controller": bool(scalar(data, "terrain_known_to_controller", False)),
        "v_ref": float(scalar(data, "v_ref", 0.0)),
        "steps": int(metrics["steps"]),
        "final_reason": metrics["final_reason"],
        "success": bool(metrics["success"]),
        "max_abs_roll_deg": float(np.rad2deg(metrics["max_abs_roll_rad"])),
        "max_abs_support_roll_deg": float(np.rad2deg(metrics["max_abs_support_roll_rad"])),
        "max_abs_terrain_diff_m": float(metrics["max_abs_terrain_diff_m"]),
        "max_abs_leg_diff_m": float(metrics["max_abs_leg_diff_m"]),
        "terrain_comp_rmse_m": float(metrics["terrain_comp_rmse_m"]),
        "max_abs_theta_rad": float(metrics["max_abs_theta_rad"]),
        "max_abs_phi_rad": float(metrics["max_abs_phi_rad"]),
        "max_abs_x_m": float(metrics["max_abs_x_m"]),
        "action_energy": float(metrics["action_energy"]),
        "file": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_adaptive_dirs()
    paths = sorted((ADAPTIVE_DIR / "pd").glob("terrain_adaptive_*.npz"))
    paths += sorted((ADAPTIVE_DIR / "mpc").glob("terrain_adaptive_*.npz"))
    paths += sorted((ADAPTIVE_DIR / "final").rglob("terrain_adaptive_*.npz"))

    seen: set[str] = set()
    rows = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        row = summarize_file(path)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: (str(r["terrain_mode"]), str(r["controller"]), str(r["file"])))
    out = Path(args.out) if args.out else ADAPTIVE_DIR / "metrics" / "terrain_adaptive_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No adaptive terrain result files found.")
        return

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={out.resolve()}")
    print("controller, terrain, known, success, reason, max_roll, max_support_roll, max_leg_diff, comp_rmse")
    for row in rows:
        print(
            f"{row['controller']}, terrain={row['terrain_mode']}, "
            f"known={row['terrain_known_to_controller']}, success={row['success']}, reason={row['final_reason']}, "
            f"max_roll={row['max_abs_roll_deg']:.3f}deg, "
            f"max_support_roll={row['max_abs_support_roll_deg']:.3f}deg, "
            f"max_leg_diff={row['max_abs_leg_diff_m']:.4f}m, "
            f"comp_rmse={row['terrain_comp_rmse_m']:.5f}m"
        )


if __name__ == "__main__":
    main()
