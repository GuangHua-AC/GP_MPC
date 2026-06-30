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


def _scalar(data, key: str, default=np.nan):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    return value.item() if hasattr(value, "item") else value


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _terrain_diff_arrays(data, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    terrain_diffs = np.asarray(data["terrain_diffs"], dtype=float).reshape(-1) if "terrain_diffs" in data.files else np.zeros(len(states))
    leg_refs = np.asarray(data["leg_diff_refs"], dtype=float).reshape(-1) if "leg_diff_refs" in data.files else -terrain_diffs
    leg_values = np.asarray(data["leg_diff_values"], dtype=float).reshape(-1) if "leg_diff_values" in data.files else states[:, 12]
    return terrain_diffs, leg_refs, leg_values


def load_result(method: str, path: Path) -> dict | None:
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
    terrain_diffs, leg_refs, leg_values = _terrain_diff_arrays(data, states)
    n = min(len(states), len(actions), len(leg_refs), len(leg_values))
    leg_errors = leg_values[:n] - leg_refs[:n]
    action_norm = np.linalg.norm(actions[:n], axis=1)
    support_rolls = np.asarray(data["support_roll_values"], dtype=float).reshape(-1) if "support_roll_values" in data.files else (
        np.asarray(data["support_rolls"], dtype=float).reshape(-1) if "support_rolls" in data.files else np.zeros(n)
    )
    final_reason = _as_str(_scalar(data, "final_reason", "unknown"))
    if final_reason == "unknown" and len(states) >= 1200:
        final_reason = "max_steps"
    return {
        "method": method,
        "path": str(path),
        "states": states,
        "actions": actions,
        "dt": float(_scalar(data, "dt", 0.005)),
        "terrain_mode": _as_str(_scalar(data, "terrain_mode", "left_obstacle")),
        "obstacle_height": float(_scalar(data, "obstacle_height", np.nan)),
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
        "uncertainty_weight": float(_scalar(data, "uncertainty_weight", np.nan)),
        "chance_weight": float(_scalar(data, "chance_weight", np.nan)),
        "guide_weight": float(_scalar(data, "guide_weight", np.nan)),
        "terminal_weight": float(_scalar(data, "terminal_weight", np.nan)),
        "k_sigma": float(_scalar(data, "k_sigma", np.nan)),
        "seed": int(_scalar(data, "seed", -1)) if np.isfinite(float(_scalar(data, "seed", -1))) else -1,
    }


def newest_pmpc_result() -> Path | None:
    candidates = sorted((OUTPUT_DIR / "terrain" / "pmpc").glob("terrain_pmpc_*_gp_pmpc.npz"))
    if not candidates:
        candidates = sorted((OUTPUT_DIR / "terrain" / "pmpc").glob("terrain_pmpc_*.npz"))
    return candidates[-1] if candidates else None


def main() -> None:
    pmpc_path = newest_pmpc_result()
    specs = [
        ("PD/VMC-terrain", OUTPUT_DIR / "terrain" / "pd" / "terrain_left_obstacle_v0p15_pd.npz"),
    ]
    if pmpc_path is not None:
        specs.append(("GP-PMPC", pmpc_path))
    else:
        print(f"missing_GP-PMPC={OUTPUT_DIR / 'terrain' / 'pmpc'}")

    results = [r for method, path in specs if (r := load_result(method, path)) is not None]
    if not results:
        print("no terrain PMPC compare results available")
        return

    plt = _require_matplotlib()
    fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=False)
    for res in results:
        states = res["states"]
        n = min(len(states), len(res["leg_diff_refs"]), len(res["leg_diff_values"]), len(res["action_norm"]))
        t = np.arange(len(states)) * res["dt"]
        label = res["method"]
        axes[0].plot(t[:n], res["leg_diff_values"][:n], label=f"{label} leg_diff")
        axes[0].plot(t[:n], res["leg_diff_refs"][:n], "--", alpha=0.75, label=f"{label} ref")
        axes[1].plot(t[:n], res["leg_diff_errors"][:n], label=label)
        axes[2].plot(t, states[:, 8], label=label)
        axes[3].plot(t, states[:, 0], label=label)
        axes[4].plot(t, states[:, 4], label=label)
        axes[5].plot(t, states[:, 2], label=label)
        axes[6].plot(t[:n], res["action_norm"][:n], label=label)

    labels = ["leg_diff_ref vs leg_diff / m", "leg_diff_error / m", "roll / rad", "theta / rad", "phi / rad", "x / m", "action norm"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time / s")
    fig.suptitle("Terrain GP-PMPC Compare")
    fig.tight_layout()
    out_fig = OUTPUT_DIR / "terrain" / "figures" / "terrain_pmpc_compare.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    out_csv = OUTPUT_DIR / "terrain" / "metrics" / "terrain_pmpc_compare.csv"
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
        "uncertainty_weight",
        "chance_weight",
        "guide_weight",
        "terminal_weight",
        "k_sigma",
        "seed",
        "path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({key: res[key] for key in fieldnames})

    print(f"saved_figure={out_fig}")
    print(f"saved_metrics={out_csv}")
    print("method,steps,final_reason,max_abs_leg_diff_error,max_abs_roll,max_abs_theta,max_abs_phi,max_abs_x")
    for res in results:
        print(
            f"{res['method']},{res['steps']},{res['final_reason']},"
            f"{res['max_abs_leg_diff_error']:.5f},{res['max_abs_roll']:.5f},"
            f"{res['max_abs_theta']:.5f},{res['max_abs_phi']:.5f},{res['max_abs_x']:.5f}"
        )


if __name__ == "__main__":
    main()
