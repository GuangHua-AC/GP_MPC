from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from wheel_legged.utils.paths import OUTPUT_DIR


TUNING_DIR = OUTPUT_DIR / "terrain" / "pmpc" / "tuning"


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def scalar(data, key: str, default=np.nan):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    return value.item() if hasattr(value, "item") else value


def load_result(path: Path) -> dict | None:
    data = np.load(path, allow_pickle=True)
    if "states" not in data.files or "actions" not in data.files:
        print(f"skip={path} missing states/actions")
        return None
    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    if len(states) == 0 or len(actions) == 0:
        print(f"skip={path} empty rollout")
        return None
    leg_errors = np.asarray(data["leg_diff_errors"], dtype=float).reshape(-1) if "leg_diff_errors" in data.files else np.zeros(len(states))
    support_rolls = np.asarray(data["support_roll_values"], dtype=float).reshape(-1) if "support_roll_values" in data.files else np.zeros(len(states))
    action_norm = np.linalg.norm(actions, axis=1)
    return {
        "method": path.stem,
        "path": str(path),
        "states": states,
        "actions": actions,
        "dt": float(scalar(data, "dt", 0.005)),
        "leg_diff_errors": leg_errors,
        "support_rolls": support_rolls,
        "action_norm": action_norm,
        "steps": int(len(states)),
        "final_reason": str(scalar(data, "final_reason", "unknown")),
        "horizon": int(scalar(data, "horizon", -1)),
        "candidates": int(scalar(data, "candidates", -1)),
        "uncertainty_weight": float(scalar(data, "uncertainty_weight", np.nan)),
        "chance_weight": float(scalar(data, "chance_weight", np.nan)),
        "guide_weight": float(scalar(data, "guide_weight", np.nan)),
        "noise_scale": float(scalar(data, "noise_scale", np.nan)),
        "random_fraction": float(scalar(data, "random_fraction", np.nan)),
        "max_abs_leg_diff_error": float(np.max(np.abs(leg_errors))) if len(leg_errors) else 0.0,
        "max_abs_roll": float(np.max(np.abs(states[:, 8]))),
        "max_abs_support_roll": float(np.max(np.abs(support_rolls))) if len(support_rolls) else 0.0,
        "max_abs_theta": float(np.max(np.abs(states[:, 0]))),
        "max_abs_phi": float(np.max(np.abs(states[:, 4]))),
        "max_abs_x": float(np.max(np.abs(states[:, 2]))),
        "mean_action_norm": float(np.mean(action_norm)),
        "max_action_norm": float(np.max(action_norm)),
        "mean_plan_time_sec": float(scalar(data, "mean_plan_time_sec", np.nan)),
        "max_plan_time_sec": float(scalar(data, "max_plan_time_sec", np.nan)),
        "total_runtime_sec": float(scalar(data, "total_runtime_sec", np.nan)),
        "notes": "",
    }


def main() -> None:
    files = sorted(TUNING_DIR.glob("*.npz"))
    if not files:
        print(f"no_tuning_results={TUNING_DIR}")
        return
    results = [r for path in files if (r := load_result(path)) is not None]
    if not results:
        print("no valid tuning results")
        return

    out_csv = OUTPUT_DIR / "terrain" / "metrics" / "terrain_pmpc_tuning.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "steps",
        "final_reason",
        "horizon",
        "candidates",
        "uncertainty_weight",
        "chance_weight",
        "guide_weight",
        "noise_scale",
        "random_fraction",
        "max_abs_leg_diff_error",
        "max_abs_roll",
        "max_abs_support_roll",
        "max_abs_theta",
        "max_abs_phi",
        "max_abs_x",
        "mean_action_norm",
        "max_action_norm",
        "mean_plan_time_sec",
        "max_plan_time_sec",
        "total_runtime_sec",
        "notes",
        "path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({key: res[key] for key in fieldnames})

    plt = _require_matplotlib()
    fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=False)
    for res in results:
        states = res["states"]
        n = min(len(states), len(res["leg_diff_errors"]), len(res["support_rolls"]), len(res["action_norm"]))
        t = np.arange(len(states)) * res["dt"]
        label = res["method"].split("_terrain_")[0]
        axes[0].plot(t, states[:, 8], label=label)
        axes[1].plot(t[:n], res["support_rolls"][:n], label=label)
        axes[2].plot(t[:n], res["leg_diff_errors"][:n], label=label)
        axes[3].plot(t, states[:, 0], label=label)
        axes[4].plot(t, states[:, 4], label=label)
        axes[5].plot(t, states[:, 2], label=label)
        axes[6].plot(t[:n], res["action_norm"][:n], label=label)
    labels = ["roll / rad", "support_roll / rad", "leg_diff_error / m", "theta / rad", "phi / rad", "x / m", "action_norm"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("time / s")
    fig.suptitle("Terrain GP-PMPC Tuning")
    fig.tight_layout()
    out_fig = OUTPUT_DIR / "terrain" / "figures" / "terrain_pmpc_tuning.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    best = sorted(results, key=lambda r: (r["final_reason"] != "max_steps", r["max_abs_roll"], r["max_abs_leg_diff_error"]))[0]
    print(f"saved_metrics={out_csv}")
    print(f"saved_figure={out_fig}")
    print("method,steps,final_reason,max_abs_leg_diff_error,max_abs_roll,max_abs_support_roll,max_abs_theta,max_abs_phi,max_abs_x,mean_plan_time_sec,total_runtime_sec")
    for res in results:
        print(
            f"{res['method']},{res['steps']},{res['final_reason']},{res['max_abs_leg_diff_error']:.5f},"
            f"{res['max_abs_roll']:.5f},{res['max_abs_support_roll']:.5f},{res['max_abs_theta']:.5f},"
            f"{res['max_abs_phi']:.5f},{res['max_abs_x']:.5f},{res['mean_plan_time_sec']:.5f},{res['total_runtime_sec']:.3f}"
        )
    print(f"best={best['method']}")


if __name__ == "__main__":
    main()
