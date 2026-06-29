from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

JUMP_DIR = Path(__file__).resolve().parent
if str(JUMP_DIR) not in sys.path:
    sys.path.insert(0, str(JUMP_DIR))

from jump_2d_pitch_model import Jump2DPitchInitialState, Jump2DPitchModel
from jump_params import JumpParams


def _idx_at_or_after(t: np.ndarray, event_time: float) -> int | None:
    if not np.isfinite(event_time) or t.size == 0:
        return None
    idx = int(np.searchsorted(t, event_time, side="left"))
    if idx >= t.size:
        return None
    return idx


def compute_pitch_metrics(
    result: dict,
    params: JumpParams,
    theta0_deg: float,
    theta_dot0_deg_s: float,
) -> dict[str, float | bool | str]:
    t = np.asarray(result["t"], dtype=float)
    theta = np.asarray(result["theta"], dtype=float)
    theta_dot = np.asarray(result["theta_dot"], dtype=float)
    max_height = float(result["max_height"][0])
    h_target = float(params.h_target)
    takeoff_time = float(result["takeoff_time"][0])
    landing_time = float(result["landing_time"][0])
    flight_time = landing_time - takeoff_time if np.isfinite(takeoff_time) and np.isfinite(landing_time) else np.nan

    landing_idx = _idx_at_or_after(t, landing_time)
    landing_abs_theta_deg = (
        abs(float(np.rad2deg(theta[landing_idx]))) if landing_idx is not None else float("nan")
    )
    recover_abs_theta_deg = abs(float(np.rad2deg(theta[-1]))) if theta.size else float("nan")
    height_error_ratio = abs(max_height - h_target) / max(h_target, 1e-9)
    pitch_sat = float(result["pitch_torque_saturation_ratio"][0])

    reasons: list[str] = []
    if not bool(result["vertical_success"][0]):
        reasons.append("vertical_not_finished")
    if not np.isfinite(takeoff_time):
        reasons.append("missing_takeoff_time")
    if not np.isfinite(landing_time):
        reasons.append("missing_landing_time")
    if height_error_ratio >= 0.10:
        reasons.append("height_error_ratio_ge_10pct")
    if float(np.max(np.abs(np.rad2deg(theta)))) >= 15.0:
        reasons.append("max_abs_theta_ge_15deg")
    if np.isfinite(landing_abs_theta_deg) and landing_abs_theta_deg >= 10.0:
        reasons.append("landing_abs_theta_ge_10deg")
    if np.isfinite(recover_abs_theta_deg) and recover_abs_theta_deg >= 3.0:
        reasons.append("recover_abs_theta_ge_3deg")
    if pitch_sat >= 0.10:
        reasons.append("pitch_torque_saturation_ratio_ge_10pct")

    success = len(reasons) == 0
    return {
        "h_target": h_target,
        "theta0_deg": float(theta0_deg),
        "theta_dot0_deg_s": float(theta_dot0_deg_s),
        "max_height": max_height,
        "height_error_ratio": float(height_error_ratio),
        "takeoff_time": takeoff_time,
        "landing_time": landing_time,
        "flight_time": float(flight_time),
        "max_abs_theta_deg": float(np.max(np.abs(np.rad2deg(theta)))),
        "landing_abs_theta_deg": landing_abs_theta_deg,
        "recover_abs_theta_deg": recover_abs_theta_deg,
        "max_abs_theta_dot_deg_s": float(np.max(np.abs(np.rad2deg(theta_dot)))),
        "peak_force": float(result["peak_force"][0]),
        "peak_pitch_torque": float(result["peak_pitch_torque"][0]),
        "pitch_torque_saturation_ratio": pitch_sat,
        "success": success,
        "fail_reason": "ok" if success else ";".join(dict.fromkeys(reasons)),
    }


def save_result(
    out_path: Path,
    result: dict,
    params: JumpParams,
    theta0_deg: float,
    theta_dot0_deg_s: float,
    metrics: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload.update(params.to_npz_payload())
    payload.update(
        {
            "theta0_deg": np.array([theta0_deg]),
            "theta_dot0_deg_s": np.array([theta_dot0_deg_s]),
            "success": np.array([bool(metrics["success"])]),
            "fail_reason": np.array([str(metrics["fail_reason"])]),
        }
    )
    for key, value in metrics.items():
        if key in payload:
            continue
        if isinstance(value, (bool, np.bool_)):
            payload[key] = np.array([bool(value)])
        elif isinstance(value, str):
            payload[key] = np.array([value])
        else:
            payload[key] = np.array([float(value)])
    np.savez(out_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-target", type=float, default=0.03)
    parser.add_argument("--theta0-deg", type=float, default=3.0)
    parser.add_argument("--theta-dot0-deg-s", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--out", default="outputs/jump/npz/jump_2d_pitch_smoke.npz")
    args = parser.parse_args()

    params = JumpParams(h_target=args.h_target, t_final=args.t_final)
    initial = Jump2DPitchInitialState(
        theta=float(np.deg2rad(args.theta0_deg)),
        theta_dot=float(np.deg2rad(args.theta_dot0_deg_s)),
    )
    result = Jump2DPitchModel(params, initial).run()
    metrics = compute_pitch_metrics(result, params, args.theta0_deg, args.theta_dot0_deg_s)
    out = Path(args.out)
    save_result(out, result, params, args.theta0_deg, args.theta_dot0_deg_s, metrics)

    print("task=jump_2d_pitch_smoke")
    print(f"h_target={metrics['h_target']:.4f} m")
    print(f"theta0_deg={metrics['theta0_deg']:.3f}")
    print(f"theta_dot0_deg_s={metrics['theta_dot0_deg_s']:.3f}")
    print(f"max_height={metrics['max_height']:.5f} m")
    print(f"height_error_ratio={metrics['height_error_ratio']:.4f}")
    print(f"takeoff_time={metrics['takeoff_time']:.4f} s")
    print(f"landing_time={metrics['landing_time']:.4f} s")
    print(f"max_abs_theta_deg={metrics['max_abs_theta_deg']:.3f}")
    print(f"landing_abs_theta_deg={metrics['landing_abs_theta_deg']:.3f}")
    print(f"recover_abs_theta_deg={metrics['recover_abs_theta_deg']:.3f}")
    print(f"peak_force={metrics['peak_force']:.3f} N")
    print(f"peak_pitch_torque={metrics['peak_pitch_torque']:.3f} Nm")
    print(f"pitch_torque_saturation_ratio={metrics['pitch_torque_saturation_ratio']:.4f}")
    print(f"success={metrics['success']} reason={metrics['fail_reason']}")
    print(f"saved={out.resolve()}")


if __name__ == "__main__":
    main()
