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


def _infer_reason(data, states: np.ndarray, steps: int) -> str:
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


def _compute_l0_from_alpha(states: np.ndarray) -> np.ndarray:
    return 0.30 + 0.10 * np.sin(states[:, 10])


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
    l0s = np.asarray(data["L0s"], dtype=float).reshape(-1) if "L0s" in data.files else _compute_l0_from_alpha(states)
    l0_ref_value = float(_scalar(data, "L0_ref", _scalar(data, "high", _scalar(data, "low", 0.34))))
    if "L0_refs" in data.files:
        l0_refs = np.asarray(data["L0_refs"], dtype=float).reshape(-1)
    else:
        l0_refs = np.full(len(states), l0_ref_value)
    n = min(len(states), len(l0s), len(l0_refs))
    l0_err = l0s[:n] - l0_refs[:n] if n else np.asarray([0.0])
    action_norm = np.linalg.norm(actions, axis=1)
    steps = int(len(states))
    return {
        "method": method,
        "path": str(path),
        "states": states,
        "actions": actions,
        "L0s": l0s,
        "L0_refs": l0_refs,
        "dt": float(_scalar(data, "dt", 0.005)),
        "steps": steps,
        "final_reason": _infer_reason(data, states, steps),
        "L0_ref": l0_ref_value,
        "final_L0": float(l0s[n - 1]) if n else 0.0,
        "final_L0_error": float(l0_err[-1]) if n else 0.0,
        "max_abs_L0_error": float(np.max(np.abs(l0_err))) if n else 0.0,
        "max_abs_theta": float(np.max(np.abs(states[:, 0]))),
        "max_abs_phi": float(np.max(np.abs(states[:, 4]))),
        "max_abs_x": float(np.max(np.abs(states[:, 2]))),
        "mean_action_norm": float(np.mean(action_norm)),
        "max_action_norm": float(np.max(action_norm)),
        "uncertainty_weight": float(_scalar(data, "uncertainty_weight", np.nan)),
        "chance_weight": float(_scalar(data, "chance_weight", np.nan)),
        "guide_weight": float(_scalar(data, "guide_weight", np.nan)),
        "terminal_weight": float(_scalar(data, "terminal_weight", np.nan)),
        "k_sigma": float(_scalar(data, "k_sigma", np.nan)),
        "seed": int(_scalar(data, "seed", -1)) if np.isfinite(float(_scalar(data, "seed", -1))) else -1,
    }


def _latest_pmpc() -> Path | None:
    paths = sorted((OUTPUT_DIR / "height" / "pmpc").glob("height_pmpc_L0ref0p34_*.npz"))
    return paths[-1] if paths else None


def main() -> None:
    specs = [
        (
            "PD-height fixed",
            OUTPUT_DIR / "height" / "pd" / "height_fixed_L0p34_to_0p34_v0p15_pd.npz",
        ),
    ]
    latest = _latest_pmpc()
    if latest is not None:
        specs.append(("GP-PMPC height", latest))
    else:
        print(f"missing_GP-PMPC={OUTPUT_DIR / 'height' / 'pmpc'}")

    results = [row for method, path in specs if (row := load_result(method, path)) is not None]
    if not results:
        print("no height PMPC comparison results available")
        return

    plt = _require_matplotlib()
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=False)
    for res in results:
        states = res["states"]
        actions = res["actions"]
        l0s = res["L0s"]
        l0_refs = res["L0_refs"]
        n = min(len(states), len(l0s), len(l0_refs))
        t = np.arange(len(states)) * res["dt"]
        label = res["method"]
        axes[0].plot(t[:n], l0s[:n], label=f"{label} L0")
        axes[0].plot(t[:n], l0_refs[:n], "--", label=f"{label} L0_ref")
        axes[1].plot(t[:n], l0s[:n] - l0_refs[:n], label=label)
        axes[2].plot(t, states[:, 0], label=label)
        axes[3].plot(t, states[:, 4], label=label)
        axes[4].plot(t, states[:, 2], label=label)
        axes[5].plot(t, np.linalg.norm(actions, axis=1), label=label)
    labels = ["L0 / m", "L0_error / m", "theta / rad", "phi / rad", "x / m", "action norm"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time / s")
    fig.suptitle("Height GP-PMPC Comparison")
    fig.tight_layout()

    out_fig = OUTPUT_DIR / "height" / "figures" / "height_pmpc_compare.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    out_csv = OUTPUT_DIR / "height" / "metrics" / "height_pmpc_compare.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "steps",
        "final_reason",
        "L0_ref",
        "final_L0",
        "final_L0_error",
        "max_abs_L0_error",
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
    print("method,steps,final_reason,L0_ref,final_L0_error,max_abs_L0_error,max_abs_theta,max_abs_phi,max_abs_x")
    for res in results:
        print(
            f"{res['method']},{res['steps']},{res['final_reason']},"
            f"{res['L0_ref']:.3f},{res['final_L0_error']:.5f},{res['max_abs_L0_error']:.5f},"
            f"{res['max_abs_theta']:.4f},{res['max_abs_phi']:.4f},{res['max_abs_x']:.4f}"
        )


if __name__ == "__main__":
    main()
