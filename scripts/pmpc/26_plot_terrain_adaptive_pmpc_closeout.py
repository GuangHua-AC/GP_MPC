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


ADAPTIVE_DIR = OUTPUT_DIR / "terrain_adaptive"


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
        print(f"skip_{method}=empty result in {path}")
        return None
    support = np.asarray(data["support_roll_values"], dtype=float).reshape(-1) if "support_roll_values" in data.files else (
        np.asarray(data["support_rolls"], dtype=float).reshape(-1) if "support_rolls" in data.files else np.zeros(len(states))
    )
    leg_adapt = np.asarray(data["leg_diff_adapt_values"], dtype=float).reshape(-1) if "leg_diff_adapt_values" in data.files else (
        np.asarray(data["leg_diff_refs"], dtype=float).reshape(-1) if "leg_diff_refs" in data.files else np.zeros(len(states))
    )
    n = min(len(states), len(actions), len(support), len(leg_adapt))
    action_norm = np.linalg.norm(actions[:n], axis=1)
    final_reason = as_text(scalar(data, "final_reason", "unknown"))
    if final_reason == "unknown" and len(states) >= 1200:
        final_reason = "max_steps"
    return {
        "method": method,
        "steps": int(len(states)),
        "final_reason": final_reason,
        "terrain_mode": as_text(scalar(data, "terrain_mode", "left_obstacle")),
        "obstacle_height": float(scalar(data, "obstacle_height", np.nan)),
        "adaptive_gain": float(scalar(data, "adaptive_gain", np.nan)),
        "adaptive_limit": float(scalar(data, "adaptive_limit", np.nan)),
        "model_source": as_text(scalar(data, "model_source", "")),
        "max_abs_roll": float(np.max(np.abs(states[:, 8]))),
        "max_abs_support_roll": float(np.max(np.abs(support))) if len(support) else 0.0,
        "max_abs_theta": float(np.max(np.abs(states[:, 0]))),
        "max_abs_phi": float(np.max(np.abs(states[:, 4]))),
        "max_abs_x": float(np.max(np.abs(states[:, 2]))),
        "max_abs_leg_diff": float(np.max(np.abs(states[:, 12]))),
        "mean_action_norm": float(np.mean(action_norm)) if len(action_norm) else 0.0,
        "max_action_norm": float(np.max(action_norm)) if len(action_norm) else 0.0,
        "mean_plan_time_sec": float(scalar(data, "mean_plan_time_sec", np.nan)),
        "total_runtime_sec": float(scalar(data, "total_runtime_sec", np.nan)),
        "uncertainty_weight": float(scalar(data, "uncertainty_weight", np.nan)),
        "chance_weight": float(scalar(data, "chance_weight", np.nan)),
        "guide_weight": float(scalar(data, "guide_weight", np.nan)),
        "terminal_weight": float(scalar(data, "terminal_weight", np.nan)),
        "k_sigma": float(scalar(data, "k_sigma", np.nan)),
        "noise_scale": float(scalar(data, "noise_scale", np.nan)),
        "random_fraction": float(scalar(data, "random_fraction", np.nan)),
        "seed": int(scalar(data, "seed", -1)) if np.isfinite(float(scalar(data, "seed", -1))) else -1,
        "notes": notes,
        "path": str(path),
        "states": states,
        "actions": actions,
        "support_rolls": support,
        "leg_diff_adapt_values": leg_adapt,
        "action_norm": action_norm,
        "dt": float(scalar(data, "dt", 0.005)),
    }


def latest(pattern: str) -> Path | None:
    matches = sorted(Path().glob(pattern))
    return matches[-1] if matches else None


def specs() -> list[tuple[str, Path | None, str]]:
    items: list[tuple[str, Path | None, str]] = [
        ("Known terrain GP-PMPC", latest("outputs/terrain/pmpc/tuning/H_chance200_guide50_*seed0.npz"), "Known terrain reference, not adaptive."),
        ("Adaptive PD/VMC", OUTPUT_DIR / "terrain_adaptive" / "pd" / "terrain_adaptive_left_obstacle_v0p15_adaptive_pd.npz", "Existing blind adaptive PD/VMC baseline."),
    ]
    for path in sorted((ADAPTIVE_DIR / "pmpc").glob("terrain_adaptive_pmpc_*seed*.npz")):
        seed = int(scalar(np.load(path, allow_pickle=True), "seed", -1))
        items.append((f"Adaptive GP-PMPC seed{seed}", path, "Recommended adaptive GP-PMPC multi-seed run."))
    for path in sorted((ADAPTIVE_DIR / "pmpc" / "ablation").glob("*.npz")):
        items.append((f"Ablation {path.stem.split('_terrain_adaptive_')[0]}", path, "Adaptive gain/limit ablation."))
    return items


def main() -> None:
    results = [r for method, path, notes in specs() if path is not None and (r := load_result(method, path, notes)) is not None]
    if not results:
        print("no adaptive closeout results")
        return
    plt = _require_matplotlib()
    fig, axes = plt.subplots(7, 1, figsize=(13, 17), sharex=False)
    for res in results:
        states = res["states"]
        n = min(len(states), len(res["support_rolls"]), len(res["leg_diff_adapt_values"]), len(res["action_norm"]))
        t = np.arange(len(states)) * res["dt"]
        label = res["method"]
        axes[0].plot(t, states[:, 8], label=label)
        axes[1].plot(t[:n], res["support_rolls"][:n], label=label)
        axes[2].plot(t, states[:, 12], label=f"{label} leg")
        axes[2].plot(t[:n], res["leg_diff_adapt_values"][:n], "--", alpha=0.65, label=f"{label} adapt")
        axes[3].plot(t, states[:, 0], label=label)
        axes[4].plot(t, states[:, 4], label=label)
        axes[5].plot(t, states[:, 2], label=label)
        axes[6].plot(t[:n], res["action_norm"][:n], label=label)
    labels = ["roll / rad", "support_roll / rad", "leg_diff / leg_diff_adapt / m", "theta / rad", "phi / rad", "x / m", "action_norm"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=6)
    axes[-1].set_xlabel("time / s")
    fig.suptitle("Terrain Adaptive GP-PMPC Closeout")
    fig.tight_layout()
    out_fig = ADAPTIVE_DIR / "figures" / "terrain_adaptive_pmpc_closeout.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    out_csv = ADAPTIVE_DIR / "metrics" / "terrain_adaptive_pmpc_closeout.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "steps",
        "final_reason",
        "terrain_mode",
        "obstacle_height",
        "adaptive_gain",
        "adaptive_limit",
        "model_source",
        "max_abs_roll",
        "max_abs_support_roll",
        "max_abs_theta",
        "max_abs_phi",
        "max_abs_x",
        "max_abs_leg_diff",
        "mean_action_norm",
        "max_action_norm",
        "mean_plan_time_sec",
        "total_runtime_sec",
        "uncertainty_weight",
        "chance_weight",
        "guide_weight",
        "terminal_weight",
        "k_sigma",
        "noise_scale",
        "random_fraction",
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
    print("method,seed,steps,final_reason,max_abs_roll,max_abs_support_roll,max_abs_theta,max_abs_phi,max_abs_x,max_abs_leg_diff,mean_plan_time_sec,total_runtime_sec")
    for res in results:
        print(
            f"{res['method']},{res['seed']},{res['steps']},{res['final_reason']},{res['max_abs_roll']:.5f},"
            f"{res['max_abs_support_roll']:.5f},{res['max_abs_theta']:.5f},{res['max_abs_phi']:.5f},"
            f"{res['max_abs_x']:.5f},{res['max_abs_leg_diff']:.5f},{res['mean_plan_time_sec']:.5f},{res['total_runtime_sec']:.3f}"
        )


if __name__ == "__main__":
    main()
