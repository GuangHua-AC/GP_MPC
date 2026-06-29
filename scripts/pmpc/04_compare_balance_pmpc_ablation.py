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


def _infer_final_reason(data, states: np.ndarray, steps: int) -> str:
    reason = str(_scalar(data, "final_reason", "unknown"))
    if reason != "unknown":
        return reason
    if abs(states[-1, 0]) >= 0.79:
        return "fall_theta"
    if abs(states[-1, 4]) >= 0.79:
        return "fall_phi"
    if steps >= 1200:
        return "max_steps"
    return "unknown"


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if "states" not in data.files or "actions" not in data.files:
        return None
    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    if len(states) == 0:
        return None
    action_norm = np.linalg.norm(actions, axis=1)
    return {
        "method": path.stem,
        "path": str(path),
        "states": states,
        "actions": actions,
        "dt": float(_scalar(data, "dt", 0.005)),
        "steps": int(len(states)),
        "final_reason": _infer_final_reason(data, states, int(len(states))),
        "max_abs_theta": float(np.max(np.abs(states[:, 0]))),
        "max_abs_phi": float(np.max(np.abs(states[:, 4]))),
        "max_abs_x": float(np.max(np.abs(states[:, 2]))),
        "max_abs_xdot": float(np.max(np.abs(states[:, 3]))),
        "max_action_norm": float(np.max(action_norm)),
        "mean_action_norm": float(np.mean(action_norm)),
        "uncertainty_weight": float(_scalar(data, "uncertainty_weight", np.nan)),
        "chance_weight": float(_scalar(data, "chance_weight", np.nan)),
        "guide_weight": float(_scalar(data, "guide_weight", np.nan)),
        "terminal_weight": float(_scalar(data, "terminal_weight", np.nan)),
        "k_sigma": float(_scalar(data, "k_sigma", np.nan)),
        "push_start": float(_scalar(data, "push_start", np.nan)),
    }


def main() -> None:
    ablation_dir = OUTPUT_DIR / "balance" / "pmpc" / "ablation"
    results = [r for p in sorted(ablation_dir.glob("*.npz")) if (r := load_result(p)) is not None]
    if not results:
        print(f"no ablation results found in {ablation_dir}")
        return

    plt = _require_matplotlib()
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=False)
    for res in results:
        t = np.arange(res["steps"]) * res["dt"]
        states = res["states"]
        actions = res["actions"]
        label = res["method"]
        axes[0].plot(t, states[:, 0], label=label)
        axes[1].plot(t, states[:, 4], label=label)
        axes[2].plot(t, states[:, 2], label=label)
        axes[3].plot(t, states[:, 3], label=label)
        axes[4].plot(t, np.linalg.norm(actions, axis=1), label=label)
    axes[0].set_ylabel("theta / rad")
    axes[1].set_ylabel("phi / rad")
    axes[2].set_ylabel("x / m")
    axes[3].set_ylabel("x_dot / m/s")
    axes[4].set_ylabel("action norm")
    axes[4].set_xlabel("time / s")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    out_fig = OUTPUT_DIR / "balance" / "figures" / "balance_pmpc_ablation.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    out_csv = OUTPUT_DIR / "balance" / "metrics" / "balance_pmpc_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "steps",
        "final_reason",
        "max_abs_theta",
        "max_abs_phi",
        "max_abs_x",
        "max_abs_xdot",
        "max_action_norm",
        "mean_action_norm",
        "uncertainty_weight",
        "chance_weight",
        "guide_weight",
        "terminal_weight",
        "k_sigma",
        "push_start",
        "path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({key: res[key] for key in fieldnames})

    print(f"saved_figure={out_fig}")
    print(f"saved_metrics={out_csv}")
    print("method,steps,final_reason,max_abs_theta,max_abs_phi,max_abs_x,max_abs_xdot,max_action_norm")
    for res in results:
        print(
            f"{res['method']},{res['steps']},{res['final_reason']},"
            f"{res['max_abs_theta']:.4f},{res['max_abs_phi']:.4f},"
            f"{res['max_abs_x']:.4f},{res['max_abs_xdot']:.4f},"
            f"{res['max_action_norm']:.4f}"
        )


if __name__ == "__main__":
    main()
