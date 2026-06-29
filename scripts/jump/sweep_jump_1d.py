from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

JUMP_DIR = Path(__file__).resolve().parent
if str(JUMP_DIR) not in sys.path:
    sys.path.insert(0, str(JUMP_DIR))

from jump_1d_model import Jump1DModel
from jump_params import JumpParams
from jump_phase_observer import JumpPhase
from test_jump_1d import save_result


FIELDNAMES = [
    "h_target",
    "max_height",
    "height_error",
    "height_error_ratio",
    "takeoff_time",
    "landing_time",
    "flight_time",
    "peak_force",
    "saturation_ratio",
    "vz_takeoff",
    "vz_landing",
    "h_pred_from_takeoff",
    "energy_height_error",
    "recover_height_error",
    "success",
    "failure_reason",
    "npz_file",
]


def _first_index_at_or_after(t: np.ndarray, event_time: float) -> int | None:
    if not np.isfinite(event_time) or t.size == 0:
        return None
    idx = int(np.searchsorted(t, event_time, side="left"))
    if idx >= t.size:
        return None
    return idx


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def compute_metrics(result: dict, params: JumpParams, npz_file: Path) -> dict[str, float | bool | str]:
    t = np.asarray(result["t"], dtype=float)
    z_b = np.asarray(result["z_b"], dtype=float)
    z_com_dot = np.asarray(result["z_com_dot"], dtype=float)
    force_saturated = np.asarray(result["force_saturated"], dtype=float)
    phase = np.asarray(result["phase"], dtype=int)

    h_target = float(params.h_target)
    max_height = float(result["max_height"][0])
    takeoff_time = float(result["takeoff_time"][0])
    landing_time = float(result["landing_time"][0])
    peak_force = float(result["peak_force"][0])
    model_success = bool(result["success"][0])

    takeoff_idx = _first_index_at_or_after(t, takeoff_time)
    landing_idx = _first_index_at_or_after(t, landing_time)
    vz_takeoff = float(z_com_dot[takeoff_idx]) if takeoff_idx is not None else float("nan")
    vz_landing = float(z_com_dot[landing_idx]) if landing_idx is not None else float("nan")
    flight_time = landing_time - takeoff_time if np.isfinite(takeoff_time) and np.isfinite(landing_time) else float("nan")
    h_pred = (vz_takeoff**2) / (2.0 * params.g) if np.isfinite(vz_takeoff) else float("nan")

    height_error = max_height - h_target
    height_error_ratio = abs(height_error) / max(h_target, 1e-9)
    energy_height_error = h_pred - max_height if np.isfinite(h_pred) else float("nan")
    recover_height_error = float(z_b[-1] - (params.wheel_radius + params.L_stand))
    saturation_ratio = float(np.mean(force_saturated)) if force_saturated.size else float("nan")
    phases_seen = set(int(x) for x in phase)

    reasons: list[str] = []
    if JumpPhase.FLIGHT_UP not in phases_seen:
        reasons.append("no_takeoff")
    if JumpPhase.LANDING not in phases_seen:
        reasons.append("no_landing")
    if not np.isfinite(takeoff_time):
        reasons.append("missing_takeoff_time")
    if not np.isfinite(landing_time):
        reasons.append("missing_landing_time")
    if height_error_ratio >= 0.10:
        reasons.append("height_error_ratio_ge_10pct")
    if saturation_ratio >= 0.05:
        reasons.append("saturation_ratio_ge_5pct")
    if abs(recover_height_error) >= 0.015:
        reasons.append("recover_not_converged")
    if not model_success:
        reasons.append("model_not_finished")

    success = len(reasons) == 0

    return {
        "h_target": h_target,
        "max_height": max_height,
        "height_error": height_error,
        "height_error_ratio": height_error_ratio,
        "takeoff_time": _safe_float(takeoff_time),
        "landing_time": _safe_float(landing_time),
        "flight_time": _safe_float(flight_time),
        "peak_force": peak_force,
        "saturation_ratio": saturation_ratio,
        "vz_takeoff": _safe_float(vz_takeoff),
        "vz_landing": _safe_float(vz_landing),
        "h_pred_from_takeoff": _safe_float(h_pred),
        "energy_height_error": _safe_float(energy_height_error),
        "recover_height_error": recover_height_error,
        "success": success,
        "failure_reason": "ok" if success else ";".join(dict.fromkeys(reasons)),
        "npz_file": str(npz_file.resolve()),
    }


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_sweep(rows: list[dict], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    h = np.asarray([r["h_target"] for r in rows], dtype=float)
    height_error = np.asarray([r["height_error"] for r in rows], dtype=float)
    height_error_ratio = np.asarray([r["height_error_ratio"] for r in rows], dtype=float)
    peak_force = np.asarray([r["peak_force"] for r in rows], dtype=float)
    sat = np.asarray([r["saturation_ratio"] for r in rows], dtype=float)
    vz_landing = np.asarray([r["vz_landing"] for r in rows], dtype=float)
    recover = np.asarray([r["recover_height_error"] for r in rows], dtype=float)
    success = np.asarray([bool(r["success"]) for r in rows])

    colors = np.where(success, "tab:green", "tab:red")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="0.25", linewidth=1.0)
    ax.axhline(0.10, color="tab:red", linestyle="--", linewidth=1.0, label="10% bound")
    ax.axhline(-0.10, color="tab:red", linestyle="--", linewidth=1.0)
    ax.bar(h, height_error_ratio * np.sign(height_error), width=0.006, color=colors, alpha=0.8)
    ax.set_xlabel("h_target / m")
    ax.set_ylabel("signed height error ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "jump_1d_sweep_height_error.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(h, peak_force, "o-", label="peak force")
    ax1.set_xlabel("h_target / m")
    ax1.set_ylabel("peak force / N")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.axhline(0.05, color="tab:red", linestyle="--", linewidth=1.0, label="5% saturation")
    ax2.plot(h, sat, "s--", color="tab:orange", label="saturation ratio")
    ax2.set_ylabel("saturation ratio")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "jump_1d_sweep_force.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(h, vz_landing, "o-", label="vz landing")
    ax1.set_xlabel("h_target / m")
    ax1.set_ylabel("landing vertical speed / m/s")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.axhline(0.015, color="tab:red", linestyle="--", linewidth=1.0, label="recover bound")
    ax2.axhline(-0.015, color="tab:red", linestyle="--", linewidth=1.0)
    ax2.plot(h, recover, "s--", color="tab:purple", label="recover height error")
    ax2.set_ylabel("recover height error / m")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "jump_1d_sweep_landing.png", dpi=160)
    plt.close(fig)


def parse_targets(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="0.01,0.02,0.03,0.04,0.05,0.06")
    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--npz-dir", default="outputs/jump/npz/sweep")
    parser.add_argument("--csv", default="outputs/jump/reports/jump_1d_sweep.csv")
    parser.add_argument("--fig-dir", default="outputs/jump/figures")
    args = parser.parse_args()

    rows: list[dict] = []
    npz_dir = Path(args.npz_dir)
    for h_target in parse_targets(args.targets):
        params = JumpParams(h_target=h_target, t_final=args.t_final)
        result = Jump1DModel(params).run()
        tag = f"{h_target:.3f}".replace(".", "p")
        npz_file = npz_dir / f"jump_1d_h{tag}.npz"
        save_result(npz_file, result, params)
        row = compute_metrics(result, params, npz_file)
        rows.append(row)
        print(
            "h_target={h_target:.3f} max_height={max_height:.4f} "
            "height_error_ratio={height_error_ratio:.3f} peak_force={peak_force:.1f} "
            "saturation_ratio={saturation_ratio:.3f} success={success} reason={failure_reason}".format(**row)
        )

    csv_path = Path(args.csv)
    write_csv(rows, csv_path)
    plot_sweep(rows, Path(args.fig_dir))
    print(f"saved_csv={csv_path.resolve()}")
    print(f"saved_fig_dir={Path(args.fig_dir).resolve()}")


if __name__ == "__main__":
    main()
