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
    if abs(states[-1, 2]) >= float(_scalar(data, "x_limit", 2.0)):
        return "x_out"
    if steps >= 1200:
        return "max_steps"
    return "unknown"


def _find_first(paths: list[Path], label: str) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    print(f"missing_{label}=" + " | ".join(str(path) for path in paths))
    return None


def _recommended_paths() -> list[Path]:
    pmpc_dir = OUTPUT_DIR / "balance" / "pmpc"
    top_level = sorted(pmpc_dir.glob("*Uw5_Cw20_Gw20_K2_Tw0_seed0_pushStart1*_gp_pmpc.npz"))
    return top_level + [pmpc_dir / "ablation" / "G_chance_guide_Uw5_Cw20_Gw20_Tw0_seed0.npz"]


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
    action_norm = np.linalg.norm(actions, axis=1)
    steps = int(len(states))
    return {
        "method": method,
        "path": str(path),
        "states": states,
        "actions": actions,
        "dt": float(_scalar(data, "dt", 0.005)),
        "steps": steps,
        "final_reason": _infer_reason(data, states, steps),
        "max_abs_theta": float(np.max(np.abs(states[:, 0]))),
        "max_abs_phi": float(np.max(np.abs(states[:, 4]))),
        "max_abs_x": float(np.max(np.abs(states[:, 2]))),
        "max_abs_xdot": float(np.max(np.abs(states[:, 3]))),
        "mean_action_norm": float(np.mean(action_norm)),
        "max_action_norm": float(np.max(action_norm)),
        "notes": notes,
    }


def main() -> None:
    result_specs = [
        (
            "PD archived success",
            [OUTPUT_DIR / "balance" / "final" / "results" / "pd" / "balance_external_push_30N_120ms_T1p2_Tp1p5_pd.npz"],
            "Dynamics model + PD/VMC archived successful 30N push result.",
        ),
        (
            "NN-MPC archived success",
            [OUTPUT_DIR / "balance" / "final" / "results" / "nn_mpc" / "balance_external_push_30N_120ms_T1p2_Tp1p5_nn_mpc_torch.npz"],
            "Torch NN dynamics + MPC archived successful 30N push result.",
        ),
        (
            "GP-MPC archived success",
            [OUTPUT_DIR / "balance" / "final" / "results" / "gp_mpc" / "balance_external_push_30N_120ms_T1p2_Tp1p5_gp_mpc.npz"],
            "Archived GP-MPC success, recorded with its saved rollout configuration.",
        ),
        (
            "GP-MPC mean-only failure",
            [OUTPUT_DIR / "balance" / "pmpc" / "ablation" / "A_original_gp_mpc_Uw0.npz"],
            "Original GP-MPC mean-only failure under closeout horizon=8, candidates=96.",
        ),
        (
            "GP-PMPC recommended success",
            _recommended_paths(),
            "Uw5 Cw20 Gw20 Tw0 K2 seed0, risk-aware GP-PMPC with safety-guided action regularization.",
        ),
    ]

    results = []
    for method, paths, notes in result_specs:
        path = _find_first(paths, method.replace(" ", "_").lower())
        if path is not None:
            result = load_result(method, path, notes)
            if result is not None:
                results.append(result)

    if not results:
        print("no closeout results available")
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
    fig.suptitle("Balance GP-PMPC Closeout")
    fig.tight_layout()

    out_fig = OUTPUT_DIR / "balance" / "figures" / "balance_pmpc_closeout.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    out_csv = OUTPUT_DIR / "balance" / "metrics" / "balance_pmpc_closeout.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "steps",
        "final_reason",
        "max_abs_theta",
        "max_abs_phi",
        "max_abs_x",
        "max_abs_xdot",
        "mean_action_norm",
        "max_action_norm",
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
    print("method,steps,final_reason,max_abs_theta,max_abs_phi,max_abs_x")
    for res in results:
        print(
            f"{res['method']},{res['steps']},{res['final_reason']},"
            f"{res['max_abs_theta']:.4f},{res['max_abs_phi']:.4f},{res['max_abs_x']:.4f}"
        )


if __name__ == "__main__":
    main()
