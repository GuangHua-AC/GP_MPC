from __future__ import annotations

import csv
import glob
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


def _latest(pattern: str) -> Path | None:
    files = sorted((Path(p) for p in glob.glob(pattern)), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _scalar(data, key: str, default="unknown"):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item() if hasattr(arr.reshape(-1)[0], "item") else arr.reshape(-1)[0]


def load_result(method: str, path: Path | None) -> dict | None:
    if path is None or not path.exists():
        print(f"skip {method}: result not found")
        return None
    data = np.load(path, allow_pickle=True)
    if "states" not in data.files or "actions" not in data.files:
        print(f"skip {method}: missing states/actions in {path}")
        return None
    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    if len(states) == 0:
        print(f"skip {method}: empty states in {path}")
        return None
    final_reason = str(_scalar(data, "final_reason", "unknown"))
    if final_reason == "unknown":
        final_reason = "max_steps_or_unknown"
    return {
        "method": method,
        "path": path,
        "states": states,
        "actions": actions,
        "dt": float(_scalar(data, "dt", 0.005)),
        "steps": int(len(states)),
        "max_abs_theta": float(np.max(np.abs(states[:, 0]))),
        "max_abs_phi": float(np.max(np.abs(states[:, 4]))),
        "max_abs_x": float(np.max(np.abs(states[:, 2]))),
        "final_x": float(states[-1, 2]),
        "final_reason": final_reason,
    }


def main() -> None:
    root = OUTPUT_DIR / "balance"
    candidates = [
        ("PD", _latest(str(root / "pd" / "balance_external_push_30N_120ms_T1p2_Tp1p5_pd.npz"))),
        ("NN-MPC", _latest(str(root / "mpc" / "*30N*Tp1p5*nn_mpc*.npz"))),
        ("GP-MPC", _latest(str(root / "mpc" / "*30N*Tp1p5*gp_mpc*.npz"))),
        ("GP-PMPC mean-only", _latest(str(root / "pmpc" / "*Uw0_Cw0*_gp_pmpc.npz"))),
        ("GP-PMPC risk-aware", _latest(str(root / "pmpc" / "*Uw5_Cw50*_gp_pmpc.npz"))),
    ]
    results = [r for method, path in candidates if (r := load_result(method, path)) is not None]
    if not results:
        print("no results available for comparison")
        return

    plt = _require_matplotlib()
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=False)
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
        ax.legend(loc="best")

    out_fig = root / "figures" / "balance_pmpc_compare.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    out_csv = root / "metrics" / "balance_pmpc_compare.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "steps", "max_abs_theta", "max_abs_phi", "max_abs_x", "final_x", "final_reason", "path"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({key: res[key] for key in fieldnames})

    print(f"saved_figure={out_fig}")
    print(f"saved_metrics={out_csv}")
    print("method,steps,max_abs_theta,max_abs_phi,max_abs_x,final_x,final_reason")
    for res in results:
        print(
            f"{res['method']},{res['steps']},{res['max_abs_theta']:.4f},"
            f"{res['max_abs_phi']:.4f},{res['max_abs_x']:.4f},"
            f"{res['final_x']:.4f},{res['final_reason']}"
        )


if __name__ == "__main__":
    main()
