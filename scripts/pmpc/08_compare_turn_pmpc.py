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


def _angle_error(angle: np.ndarray | float, ref: np.ndarray | float) -> np.ndarray | float:
    return (angle - ref + np.pi) % (2.0 * np.pi) - np.pi


def _infer_reason(data, states: np.ndarray, steps: int) -> str:
    reason = str(_scalar(data, "final_reason", "unknown"))
    if reason != "unknown":
        return reason
    if abs(states[-1, 0]) >= 0.79:
        return "fall_theta"
    if abs(states[-1, 4]) >= 0.79:
        return "fall_phi"
    if abs(states[-1, 8]) >= 0.79:
        return "fall_roll"
    if steps >= 1200:
        return "max_steps"
    return "unknown"


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
        print(f"skip_{method}=empty rollout in {path}")
        return None
    target_deg = float(_scalar(data, "target_deg", 30.0))
    yaw_ref = float(_scalar(data, "yaw_ref", np.deg2rad(target_deg)))
    yaw_refs = np.asarray(data["yaw_refs"], dtype=float) if "yaw_refs" in data.files else np.full(len(states), yaw_ref)
    n = min(len(states), len(yaw_refs))
    yaw_error = _angle_error(states[:n, 6], yaw_refs[:n]) if n else np.asarray([0.0])
    action_norm = np.linalg.norm(actions, axis=1)
    final_yaw_deg = float(np.rad2deg(states[-1, 6]))
    final_yaw_error_deg = float(np.rad2deg(yaw_error[-1])) if n else 0.0
    steps = int(len(states))
    return {
        "method": method,
        "path": str(path),
        "states": states,
        "actions": actions,
        "yaw_refs": yaw_refs,
        "dt": float(_scalar(data, "dt", 0.005)),
        "steps": steps,
        "final_reason": _infer_reason(data, states, steps),
        "target_deg": target_deg,
        "final_yaw_deg": final_yaw_deg,
        "final_yaw_error_deg": final_yaw_error_deg,
        "max_abs_roll": float(np.max(np.abs(states[:, 8]))),
        "max_abs_theta": float(np.max(np.abs(states[:, 0]))),
        "max_abs_phi": float(np.max(np.abs(states[:, 4]))),
        "max_abs_x": float(np.max(np.abs(states[:, 2]))),
        "mean_action_norm": float(np.mean(action_norm)),
        "max_action_norm": float(np.max(action_norm)),
    }


def _latest_pmpc() -> Path | None:
    paths = sorted((OUTPUT_DIR / "turn" / "pmpc").glob("turn_pmpc_target30deg_v0p15_*.npz"))
    return paths[-1] if paths else None


def main() -> None:
    specs = [
        (
            "PD-roll",
            OUTPUT_DIR / "turn" / "pd" / "turn_roll_target30deg_v0p15_roll00deg_pd.npz",
        ),
        (
            "PD-no-roll",
            OUTPUT_DIR / "turn" / "pd" / "turn_roll_target30deg_v0p15_roll00deg_pd_no_roll.npz",
        ),
    ]
    latest_pmpc = _latest_pmpc()
    if latest_pmpc is not None:
        specs.append(("GP-PMPC", latest_pmpc))
    else:
        print(f"missing_GP-PMPC={OUTPUT_DIR / 'turn' / 'pmpc'}")

    results = [row for method, path in specs if (row := load_result(method, path)) is not None]
    if not results:
        print("no turn PMPC comparison results available")
        return

    plt = _require_matplotlib()
    fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=False)
    for res in results:
        states = res["states"]
        actions = res["actions"]
        yaw_refs = res["yaw_refs"]
        n = min(len(states), len(yaw_refs))
        t = np.arange(len(states)) * res["dt"]
        label = res["method"]
        axes[0].plot(t, np.rad2deg(states[:, 6]), label=label)
        axes[1].plot(t[:n], np.rad2deg(_angle_error(states[:n, 6], yaw_refs[:n])), label=label)
        axes[2].plot(t, np.rad2deg(states[:, 8]), label=label)
        axes[3].plot(t, states[:, 0], label=label)
        axes[4].plot(t, states[:, 4], label=label)
        axes[5].plot(t, states[:, 2], label=label)
        axes[6].plot(t, np.linalg.norm(actions, axis=1), label=label)

    labels = ["yaw / deg", "yaw_error / deg", "roll / deg", "theta / rad", "phi / rad", "x / m", "action norm"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time / s")
    fig.suptitle("Turn GP-PMPC Comparison")
    fig.tight_layout()

    out_fig = OUTPUT_DIR / "turn" / "figures" / "turn_pmpc_compare.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    out_csv = OUTPUT_DIR / "turn" / "metrics" / "turn_pmpc_compare.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "steps",
        "final_reason",
        "target_deg",
        "final_yaw_deg",
        "final_yaw_error_deg",
        "max_abs_roll",
        "max_abs_theta",
        "max_abs_phi",
        "max_abs_x",
        "mean_action_norm",
        "max_action_norm",
        "path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({key: res[key] for key in fieldnames})

    print(f"saved_figure={out_fig}")
    print(f"saved_metrics={out_csv}")
    print("method,steps,final_reason,target_deg,final_yaw_error_deg,max_abs_roll,max_abs_theta,max_abs_phi,max_abs_x")
    for res in results:
        print(
            f"{res['method']},{res['steps']},{res['final_reason']},{res['target_deg']:.1f},"
            f"{res['final_yaw_error_deg']:.3f},{res['max_abs_roll']:.4f},"
            f"{res['max_abs_theta']:.4f},{res['max_abs_phi']:.4f},{res['max_abs_x']:.4f}"
        )


if __name__ == "__main__":
    main()
