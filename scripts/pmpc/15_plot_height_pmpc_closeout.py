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


def _compute_l0(states: np.ndarray) -> np.ndarray:
    return 0.30 + 0.10 * np.sin(states[:, 10])


def _metrics_from_arrays(l0s: np.ndarray, refs: np.ndarray, dt: float) -> dict[str, float]:
    n = min(len(l0s), len(refs))
    if n == 0:
        return {
            "max_abs_L0_error": 0.0,
            "max_abs_L0_error_after_1s": 0.0,
            "rmse_L0_error": 0.0,
            "rmse_L0_error_after_1s": 0.0,
            "final_L0_error": 0.0,
            "settling_time_2cm": np.nan,
        }
    err = l0s[:n] - refs[:n]
    start = min(n - 1, int(np.ceil(1.0 / max(dt, 1e-9))))
    err_after = err[start:]
    window = max(1, int(round(0.5 / max(dt, 1e-9))))
    settling = np.nan
    for i in range(n):
        if np.all(np.abs(err[i : min(n, i + window)]) <= 0.02):
            settling = i * dt
            break
    return {
        "max_abs_L0_error": float(np.max(np.abs(err))),
        "max_abs_L0_error_after_1s": float(np.max(np.abs(err_after))) if len(err_after) else float(np.max(np.abs(err))),
        "rmse_L0_error": float(np.sqrt(np.mean(err**2))),
        "rmse_L0_error_after_1s": float(np.sqrt(np.mean(err_after**2))) if len(err_after) else float(np.sqrt(np.mean(err**2))),
        "final_L0_error": float(err[-1]),
        "settling_time_2cm": float(settling),
    }


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


def load_result(method: str, mode: str, path: Path, notes: str) -> dict | None:
    if not path.exists():
        print(f"missing_{method}_{mode}={path}")
        return None
    data = np.load(path, allow_pickle=True)
    if "states" not in data.files or "actions" not in data.files:
        print(f"skip_{method}_{mode}=missing states/actions in {path}")
        return None
    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    if len(states) == 0 or len(actions) == 0:
        print(f"skip_{method}_{mode}=empty rollout in {path}")
        return None
    l0s = np.asarray(data["L0_values"], dtype=float).reshape(-1) if "L0_values" in data.files else (
        np.asarray(data["L0s"], dtype=float).reshape(-1) if "L0s" in data.files else _compute_l0(states)
    )
    if "L0_refs" in data.files:
        refs = np.asarray(data["L0_refs"], dtype=float).reshape(-1)
    else:
        refs = np.full(len(l0s), float(_scalar(data, "L0_ref", _scalar(data, "high", 0.34))))
    dt = float(_scalar(data, "dt", 0.005))
    metric = _metrics_from_arrays(l0s, refs, dt)
    action_norm = np.linalg.norm(actions, axis=1)
    steps = int(len(states))
    return {
        "method": method,
        "mode": mode,
        "path": str(path),
        "states": states,
        "actions": actions,
        "L0_values": l0s,
        "L0_refs": refs,
        "dt": dt,
        "steps": steps,
        "final_reason": _infer_reason(data, states, steps),
        "max_abs_L0_error": float(_scalar(data, "max_abs_L0_error", metric["max_abs_L0_error"])),
        "max_abs_L0_error_after_1s": float(_scalar(data, "max_abs_L0_error_after_1s", metric["max_abs_L0_error_after_1s"])),
        "rmse_L0_error": float(_scalar(data, "rmse_L0_error", metric["rmse_L0_error"])),
        "rmse_L0_error_after_1s": float(_scalar(data, "rmse_L0_error_after_1s", metric["rmse_L0_error_after_1s"])),
        "final_L0_error": float(_scalar(data, "final_L0_error", metric["final_L0_error"])),
        "settling_time_2cm": float(_scalar(data, "settling_time_2cm", metric["settling_time_2cm"])),
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
        "noise_scale": float(_scalar(data, "noise_scale", np.nan)),
        "random_fraction": float(_scalar(data, "random_fraction", np.nan)),
        "seed": int(_scalar(data, "seed", -1)) if np.isfinite(float(_scalar(data, "seed", -1))) else -1,
        "notes": notes,
    }


def main() -> None:
    tracking = OUTPUT_DIR / "height" / "pmpc" / "tracking"
    specs = [
        ("PD fixed", "fixed", OUTPUT_DIR / "height" / "pd" / "height_fixed_L0p34_to_0p34_v0p15_pd.npz", "Existing fixed-height PD/VMC baseline."),
        ("PD step", "step", OUTPUT_DIR / "height" / "pd" / "height_step_L0p3_to_0p34_v0p15_pd.npz", "Existing step PD/VMC baseline if available."),
        ("PD sine", "sine", OUTPUT_DIR / "height" / "pd" / "height_sine_L0p3_to_0p34_v0p15_pd.npz", "Existing sine PD/VMC baseline if available."),
        ("PMPC fixed", "fixed", tracking / "A_fixed_height_pmpc_fixed_Uw5_Cw20_Gw20_Tw0_K2_seed0.npz", "Recommended fixed PMPC."),
        ("PMPC step", "step", tracking / "B_step_height_pmpc_step_Uw5_Cw20_Gw20_Tw0_K2_seed0.npz", "Recommended step PMPC."),
        ("PMPC sine", "sine", tracking / "C_sine_height_pmpc_sine_Uw5_Cw20_Gw20_Tw0_K2_seed0.npz", "Recommended sine PMPC."),
    ]
    results = [r for method, mode, path, notes in specs if (r := load_result(method, mode, path, notes)) is not None]
    if not results:
        print("no height closeout results available")
        return

    plt = _require_matplotlib()
    fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=False)
    for res in results:
        states = res["states"]
        actions = res["actions"]
        l0s = res["L0_values"]
        refs = res["L0_refs"]
        n = min(len(states), len(l0s), len(refs))
        t = np.arange(len(states)) * res["dt"]
        label = res["method"]
        axes[0].plot(t[:n], l0s[:n], label=f"{label} L0")
        axes[0].plot(t[:n], refs[:n], "--", alpha=0.75, label=f"{label} ref")
        axes[1].plot(t[:n], l0s[:n] - refs[:n], label=label)
        axes[2].plot(t, states[:, 0], label=label)
        axes[3].plot(t, states[:, 4], label=label)
        axes[4].plot(t, states[:, 2], label=label)
        axes[5].plot(t, np.linalg.norm(actions, axis=1), label=label)
    labels = ["L0_ref vs L0 / m", "L0_error / m", "theta / rad", "phi / rad", "x / m", "action norm"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("time / s")
    fig.suptitle("Height GP-PMPC Closeout")
    fig.tight_layout()
    out_fig = OUTPUT_DIR / "height" / "figures" / "height_pmpc_closeout.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    out_csv = OUTPUT_DIR / "height" / "metrics" / "height_pmpc_closeout.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "mode",
        "steps",
        "final_reason",
        "max_abs_L0_error",
        "max_abs_L0_error_after_1s",
        "rmse_L0_error",
        "rmse_L0_error_after_1s",
        "final_L0_error",
        "settling_time_2cm",
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
    print("method,mode,steps,final_reason,max_abs_L0_error_after_1s,rmse_L0_error_after_1s,final_L0_error,max_abs_theta,max_abs_phi,max_abs_x")
    for res in results:
        print(
            f"{res['method']},{res['mode']},{res['steps']},{res['final_reason']},"
            f"{res['max_abs_L0_error_after_1s']:.5f},{res['rmse_L0_error_after_1s']:.5f},"
            f"{res['final_L0_error']:.5f},{res['max_abs_theta']:.4f},{res['max_abs_phi']:.4f},{res['max_abs_x']:.4f}"
        )


if __name__ == "__main__":
    main()
