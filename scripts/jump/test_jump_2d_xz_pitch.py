from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

JUMP_DIR = Path(__file__).resolve().parent
if str(JUMP_DIR) not in sys.path:
    sys.path.insert(0, str(JUMP_DIR))

from jump_2d_xz_pitch_model import Jump2DXZPitchInitialState, Jump2DXZPitchModel
from jump_params import JumpParams


def _idx_at_or_after(t: np.ndarray, event_time: float) -> int | None:
    if not np.isfinite(event_time) or t.size == 0:
        return None
    idx = int(np.searchsorted(t, event_time, side="left"))
    if idx >= t.size:
        return None
    return idx


def compute_xz_pitch_metrics(
    result: dict,
    params: JumpParams,
    theta0_deg: float,
    x_force_offset: float,
    x0: float,
    xdot0: float,
) -> dict[str, float | bool | str]:
    t = np.asarray(result["t"], dtype=float)
    x = np.asarray(result["x"], dtype=float)
    xdot = np.asarray(result["xdot"], dtype=float)
    theta = np.asarray(result["theta"], dtype=float)
    theta_dot = np.asarray(result["theta_dot"], dtype=float)

    h_target = float(params.h_target)
    max_height = float(result["max_height"][0])
    height_error_ratio = abs(max_height - h_target) / max(h_target, 1e-9)
    takeoff_time = float(result["takeoff_time"][0])
    landing_time = float(result["landing_time"][0])
    flight_time = landing_time - takeoff_time if np.isfinite(takeoff_time) and np.isfinite(landing_time) else np.nan
    landing_idx = _idx_at_or_after(t, landing_time)

    landing_x = float(x[landing_idx]) if landing_idx is not None else float("nan")
    landing_abs_theta_deg = abs(float(np.rad2deg(theta[landing_idx]))) if landing_idx is not None else float("nan")
    recover_abs_x = abs(float(x[-1])) if x.size else float("nan")
    recover_abs_theta_deg = abs(float(np.rad2deg(theta[-1]))) if theta.size else float("nan")
    max_abs_theta_deg = float(np.max(np.abs(np.rad2deg(theta)))) if theta.size else float("nan")
    max_abs_x = float(np.max(np.abs(x))) if x.size else float("nan")
    max_abs_xdot = float(np.max(np.abs(xdot))) if xdot.size else float("nan")
    max_abs_theta_dot_deg_s = float(np.max(np.abs(np.rad2deg(theta_dot)))) if theta_dot.size else float("nan")
    pitch_sat = float(result["pitch_torque_saturation_ratio"][0])
    wheel_sat = float(result["wheel_force_saturation_ratio"][0])

    reasons: list[str] = []
    if not bool(result["vertical_success"][0]):
        reasons.append("vertical_not_finished")
    if not np.isfinite(takeoff_time):
        reasons.append("missing_takeoff_time")
    if not np.isfinite(landing_time):
        reasons.append("missing_landing_time")
    if height_error_ratio >= 0.10:
        reasons.append("height_error_ratio_ge_10pct")
    if max_abs_theta_deg >= 15.0:
        reasons.append("max_abs_theta_ge_15deg")
    if np.isfinite(landing_abs_theta_deg) and landing_abs_theta_deg >= 10.0:
        reasons.append("landing_abs_theta_ge_10deg")
    if np.isfinite(recover_abs_theta_deg) and recover_abs_theta_deg >= 3.0:
        reasons.append("recover_abs_theta_ge_3deg")
    if max_abs_x >= 0.05:
        reasons.append("max_abs_x_ge_0p05m")
    if np.isfinite(recover_abs_x) and recover_abs_x >= 0.02:
        reasons.append("recover_abs_x_ge_0p02m")
    if pitch_sat >= 0.10:
        reasons.append("pitch_torque_saturation_ratio_ge_10pct")
    if wheel_sat >= 0.10:
        reasons.append("wheel_force_saturation_ratio_ge_10pct")

    success = len(reasons) == 0
    return {
        "h_target": h_target,
        "theta0_deg": float(theta0_deg),
        "x_force_offset": float(x_force_offset),
        "x0": float(x0),
        "xdot0": float(xdot0),
        "max_height": max_height,
        "height_error_ratio": float(height_error_ratio),
        "takeoff_time": takeoff_time,
        "landing_time": landing_time,
        "flight_time": float(flight_time),
        "max_abs_x": max_abs_x,
        "landing_x": landing_x,
        "recover_abs_x": recover_abs_x,
        "max_abs_xdot": max_abs_xdot,
        "max_abs_theta_deg": max_abs_theta_deg,
        "landing_abs_theta_deg": landing_abs_theta_deg,
        "recover_abs_theta_deg": recover_abs_theta_deg,
        "max_abs_theta_dot_deg_s": max_abs_theta_dot_deg_s,
        "peak_force_z": float(result["peak_force_z"][0]),
        "peak_force_x": float(result["peak_force_x"][0]),
        "peak_pitch_torque": float(result["peak_pitch_torque"][0]),
        "peak_tau_coupling": float(result["peak_tau_coupling"][0]),
        "pitch_torque_saturation_ratio": pitch_sat,
        "wheel_force_saturation_ratio": wheel_sat,
        "success": success,
        "fail_reason": "ok" if success else ";".join(dict.fromkeys(reasons)),
    }


def save_result(
    out_path: Path,
    result: dict,
    params: JumpParams,
    metrics: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload.update(params.to_npz_payload())
    for key, value in metrics.items():
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
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--xdot0", type=float, default=0.0)
    parser.add_argument("--x-force-offset", type=float, default=0.01)
    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--out", default="outputs/jump/npz/jump_2d_xz_pitch_smoke.npz")
    args = parser.parse_args()

    params = JumpParams(
        h_target=args.h_target,
        t_final=args.t_final,
        x_force_offset=args.x_force_offset,
    )
    initial = Jump2DXZPitchInitialState(
        x=args.x0,
        xdot=args.xdot0,
        theta=float(np.deg2rad(args.theta0_deg)),
        theta_dot=0.0,
        x_force_offset=args.x_force_offset,
    )
    result = Jump2DXZPitchModel(params, initial).run()
    metrics = compute_xz_pitch_metrics(
        result,
        params,
        args.theta0_deg,
        args.x_force_offset,
        args.x0,
        args.xdot0,
    )
    out = Path(args.out)
    save_result(out, result, params, metrics)

    print("task=jump_2d_xz_pitch_smoke")
    print(f"h_target={metrics['h_target']:.4f} m")
    print(f"theta0_deg={metrics['theta0_deg']:.3f}")
    print(f"x_force_offset={metrics['x_force_offset']:.4f} m")
    print(f"max_height={metrics['max_height']:.5f} m")
    print(f"height_error_ratio={metrics['height_error_ratio']:.4f}")
    print(f"takeoff_time={metrics['takeoff_time']:.4f} s")
    print(f"landing_time={metrics['landing_time']:.4f} s")
    print(f"max_abs_x={metrics['max_abs_x']:.5f} m")
    print(f"recover_abs_x={metrics['recover_abs_x']:.5f} m")
    print(f"max_abs_theta_deg={metrics['max_abs_theta_deg']:.3f}")
    print(f"landing_abs_theta_deg={metrics['landing_abs_theta_deg']:.3f}")
    print(f"recover_abs_theta_deg={metrics['recover_abs_theta_deg']:.3f}")
    print(f"peak_force_z={metrics['peak_force_z']:.3f} N")
    print(f"peak_force_x={metrics['peak_force_x']:.3f} N")
    print(f"peak_pitch_torque={metrics['peak_pitch_torque']:.3f} Nm")
    print(f"peak_tau_coupling={metrics['peak_tau_coupling']:.3f} Nm")
    print(f"pitch_torque_saturation_ratio={metrics['pitch_torque_saturation_ratio']:.4f}")
    print(f"wheel_force_saturation_ratio={metrics['wheel_force_saturation_ratio']:.4f}")
    print(f"success={metrics['success']} reason={metrics['fail_reason']}")
    print(f"saved={out.resolve()}")


if __name__ == "__main__":
    main()
