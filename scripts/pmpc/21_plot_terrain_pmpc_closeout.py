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


def as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_result(method: str, path: Path, notes: str) -> dict | None:
    if not path.exists():
        print(f"missing_{method}={path}")
        return None
    data = np.load(path, allow_pickle=True)
    if "states" not in data.files or "actions" not in data.files:
        print(f"skip_{method}=missing states/actions in {path}")
        return None
    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    if len(states) == 0 or len(actions) == 0:
        print(f"skip_{method}=empty rollout in {path}")
        return None

    terrain_diffs = np.asarray(data["terrain_diffs"], dtype=float).reshape(-1) if "terrain_diffs" in data.files else np.zeros(len(states))
    leg_refs = np.asarray(data["leg_diff_refs"], dtype=float).reshape(-1) if "leg_diff_refs" in data.files else -terrain_diffs
    leg_values = np.asarray(data["leg_diff_values"], dtype=float).reshape(-1) if "leg_diff_values" in data.files else states[:, 12]
    n = min(len(states), len(actions), len(leg_refs), len(leg_values))
    leg_errors = np.asarray(data["leg_diff_errors"], dtype=float).reshape(-1) if "leg_diff_errors" in data.files else leg_values[:n] - leg_refs[:n]
    support_rolls = np.asarray(data["support_roll_values"], dtype=float).reshape(-1) if "support_roll_values" in data.files else (
        np.asarray(data["support_rolls"], dtype=float).reshape(-1) if "support_rolls" in data.files else np.zeros(n)
    )
    action_norm = np.linalg.norm(actions[:n], axis=1)
    final_reason = as_text(scalar(data, "final_reason", "unknown"))
    if final_reason == "unknown" and len(states) >= 1200:
        final_reason = "max_steps"

    return {
        "method": method,
        "path": str(path),
        "notes": notes,
        "states": states,
        "actions": actions,
        "dt": float(scalar(data, "dt", 0.005)),
        "terrain_mode": as_text(scalar(data, "terrain_mode", "left_obstacle")),
        "obstacle_height": float(scalar(data, "obstacle_height", np.nan)),
        "leg_diff_refs": leg_refs,
        "leg_diff_values": leg_values,
        "leg_diff_errors": leg_errors,
        "support_rolls": support_rolls,
        "action_norm": action_norm,
        "steps": int(len(states)),
        "final_reason": final_reason,
        "max_abs_leg_diff_error": float(np.max(np.abs(leg_errors))) if len(leg_errors) else 0.0,
        "max_abs_roll": float(np.max(np.abs(states[:, 8]))),
        "max_abs_support_roll": float(np.max(np.abs(support_rolls))) if len(support_rolls) else 0.0,
        "max_abs_theta": float(np.max(np.abs(states[:, 0]))),
        "max_abs_phi": float(np.max(np.abs(states[:, 4]))),
        "max_abs_x": float(np.max(np.abs(states[:, 2]))),
        "mean_action_norm": float(np.mean(action_norm)) if len(action_norm) else 0.0,
        "max_action_norm": float(np.max(action_norm)) if len(action_norm) else 0.0,
        "mean_plan_time_sec": float(scalar(data, "mean_plan_time_sec", np.nan)),
        "max_plan_time_sec": float(scalar(data, "max_plan_time_sec", np.nan)),
        "total_runtime_sec": float(scalar(data, "total_runtime_sec", np.nan)),
        "uncertainty_weight": float(scalar(data, "uncertainty_weight", np.nan)),
        "chance_weight": float(scalar(data, "chance_weight", np.nan)),
        "guide_weight": float(scalar(data, "guide_weight", np.nan)),
        "terminal_weight": float(scalar(data, "terminal_weight", np.nan)),
        "k_sigma": float(scalar(data, "k_sigma", np.nan)),
        "noise_scale": float(scalar(data, "noise_scale", np.nan)),
        "random_fraction": float(scalar(data, "random_fraction", np.nan)),
        "horizon": int(scalar(data, "horizon", -1)) if np.isfinite(float(scalar(data, "horizon", -1))) else -1,
        "candidates": int(scalar(data, "candidates", -1)) if np.isfinite(float(scalar(data, "candidates", -1))) else -1,
        "seed": int(scalar(data, "seed", -1)) if np.isfinite(float(scalar(data, "seed", -1))) else -1,
    }


def closeout_specs() -> list[tuple[str, Path, str]]:
    terrain = OUTPUT_DIR / "terrain"
    specs = [
        ("PD terrain baseline", terrain / "pd" / "terrain_left_obstacle_v0p15_pd.npz", "Known-terrain PD/VMC baseline."),
        ("GP-PMPC v0", terrain / "pmpc" / "terrain_pmpc_left_obstacle_h0p04_x1_len0p5_T1p2_Tp1p5_Uw5_Cw20_Gw20_K2_Tw0_seed0.npz", "Stage 6.1 v0: runs full horizon but roll peak is large."),
    ]
    tuning_dir = terrain / "pmpc" / "tuning"
    for path in sorted(tuning_dir.glob("*.npz")):
        name = path.stem
        if name.startswith("H_chance200_guide50"):
            method = f"GP-PMPC recommended seed{int(scalar(np.load(path, allow_pickle=True), 'seed', -1))}"
            notes = "Recommended tuned known-terrain GP-PMPC."
        else:
            method = f"Tuning {name.split('_terrain_')[0]}"
            notes = "Terrain PMPC tuning ablation."
        specs.append((method, path, notes))
    return specs


def main() -> None:
    results = [r for method, path, notes in closeout_specs() if (r := load_result(method, path, notes)) is not None]
    if not results:
        print("no terrain closeout results available")
        return

    plt = _require_matplotlib()
    fig, axes = plt.subplots(8, 1, figsize=(13, 18), sharex=False)
    for res in results:
        states = res["states"]
        n = min(len(states), len(res["leg_diff_refs"]), len(res["leg_diff_values"]), len(res["leg_diff_errors"]), len(res["support_rolls"]), len(res["action_norm"]))
        t = np.arange(len(states)) * res["dt"]
        label = res["method"]
        axes[0].plot(t[:n], res["leg_diff_values"][:n], label=f"{label} leg")
        axes[0].plot(t[:n], res["leg_diff_refs"][:n], "--", alpha=0.65, label=f"{label} ref")
        axes[1].plot(t[:n], res["leg_diff_errors"][:n], label=label)
        axes[2].plot(t, states[:, 8], label=label)
        axes[3].plot(t[:n], res["support_rolls"][:n], label=label)
        axes[4].plot(t, states[:, 0], label=label)
        axes[5].plot(t, states[:, 4], label=label)
        axes[6].plot(t, states[:, 2], label=label)
        axes[7].plot(t[:n], res["action_norm"][:n], label=label)

    labels = [
        "leg_diff_ref vs leg_diff / m",
        "leg_diff_error / m",
        "roll / rad",
        "support_roll / rad",
        "theta / rad",
        "phi / rad",
        "x / m",
        "action_norm",
    ]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=6)
    axes[-1].set_xlabel("time / s")
    fig.suptitle("Known Terrain GP-PMPC Closeout")
    fig.tight_layout()
    out_fig = OUTPUT_DIR / "terrain" / "figures" / "terrain_pmpc_closeout.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    out_csv = OUTPUT_DIR / "terrain" / "metrics" / "terrain_pmpc_closeout.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "steps",
        "final_reason",
        "terrain_mode",
        "obstacle_height",
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
        "uncertainty_weight",
        "chance_weight",
        "guide_weight",
        "terminal_weight",
        "k_sigma",
        "noise_scale",
        "random_fraction",
        "horizon",
        "candidates",
        "seed",
        "notes",
        "path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({key: res[key] for key in fieldnames})

    print(f"saved_figure={out_fig}")
    print(f"saved_metrics={out_csv}")
    print("method,seed,steps,final_reason,max_abs_leg_diff_error,max_abs_roll,max_abs_support_roll,max_abs_theta,max_abs_phi,max_abs_x,mean_plan_time_sec,total_runtime_sec")
    for res in results:
        print(
            f"{res['method']},{res['seed']},{res['steps']},{res['final_reason']},"
            f"{res['max_abs_leg_diff_error']:.5f},{res['max_abs_roll']:.5f},{res['max_abs_support_roll']:.5f},"
            f"{res['max_abs_theta']:.5f},{res['max_abs_phi']:.5f},{res['max_abs_x']:.5f},"
            f"{res['mean_plan_time_sec']:.5f},{res['total_runtime_sec']:.3f}"
        )


if __name__ == "__main__":
    main()
