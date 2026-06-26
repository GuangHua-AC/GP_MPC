from __future__ import annotations

from pathlib import Path

import numpy as np


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def angle_error(angle: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (angle - ref + np.pi) % (2.0 * np.pi) - np.pi


def controller_label(kind: str, disable_roll_control: bool = False) -> str:
    if kind == "pd" and disable_roll_control:
        return "PD-no-roll"
    if kind == "pd":
        return "PD-roll"
    if kind == "nn_mpc":
        return "NN-MPC"
    if kind == "gp_mpc":
        return "GP-MPC"
    return kind


def roll_turn_filename(
    kind: str,
    target_deg: float,
    v_ref: float,
    roll0_deg: float,
    disable_roll_control: bool = False,
    backend: str | None = None,
) -> str:
    suffix = "_no_roll" if disable_roll_control else ""
    backend_part = f"_{backend}" if backend else ""
    return (
        f"turn_roll_target{tag_float(target_deg)}deg"
        f"_v{tag_float(v_ref)}"
        f"_roll0{tag_float(roll0_deg)}deg"
        f"_{kind}{backend_part}{suffix}.npz"
    )


def compute_metrics(
    states: np.ndarray,
    actions: np.ndarray,
    yaw_refs: np.ndarray,
    roll_refs: np.ndarray,
    final_reason: str,
) -> dict[str, float | int | str | bool]:
    if len(states) == 0:
        return {
            "steps": 0,
            "final_reason": final_reason,
            "success": False,
            "max_abs_theta_rad": 0.0,
            "max_abs_phi_rad": 0.0,
            "max_abs_yaw_error_rad": 0.0,
            "max_abs_roll_error_rad": 0.0,
            "max_abs_x_m": 0.0,
            "final_yaw_error_rad": 0.0,
            "final_roll_error_rad": 0.0,
            "action_energy": 0.0,
        }

    n = min(len(states), len(yaw_refs), len(roll_refs))
    yaw_err = angle_error(states[:n, 6], yaw_refs[:n])
    roll_err = states[:n, 8] - roll_refs[:n]
    final_yaw_err = float(yaw_err[-1]) if n else 0.0
    final_roll_err = float(roll_err[-1]) if n else 0.0
    max_theta = float(np.max(np.abs(states[:, 0])))
    max_phi = float(np.max(np.abs(states[:, 4])))
    max_yaw_err = float(np.max(np.abs(yaw_err))) if n else 0.0
    max_roll_err = float(np.max(np.abs(roll_err))) if n else 0.0
    max_x = float(np.max(np.abs(states[:, 2])))
    action_energy = float(np.sum(actions**2)) if len(actions) else 0.0
    success = (
        final_reason in {"not_done", "max_steps"}
        and abs(final_yaw_err) < np.deg2rad(3.0)
        and abs(final_roll_err) < np.deg2rad(3.0)
        and max_theta < 0.8
        and max_phi < 0.8
    )
    return {
        "steps": int(len(states)),
        "final_reason": final_reason,
        "success": bool(success),
        "max_abs_theta_rad": max_theta,
        "max_abs_phi_rad": max_phi,
        "max_abs_yaw_error_rad": max_yaw_err,
        "max_abs_roll_error_rad": max_roll_err,
        "max_abs_x_m": max_x,
        "final_yaw_error_rad": final_yaw_err,
        "final_roll_error_rad": final_roll_err,
        "action_energy": action_energy,
    }


def save_roll_turn_result(
    out: Path,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    yaw_refs: np.ndarray,
    roll_refs: np.ndarray,
    dt: float,
    target_deg: float,
    v_ref: float,
    roll0_deg: float,
    controller: str,
    final_reason: str,
    horizon: int | None = None,
    candidates: int | None = None,
    uncertainty_weight: float | None = None,
    best_costs: np.ndarray | None = None,
    uncertainty_costs: np.ndarray | None = None,
) -> dict[str, float | int | str | bool]:
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(states, actions, yaw_refs, roll_refs, final_reason)
    payload = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "yaw_refs": yaw_refs,
        "roll_refs": roll_refs,
        "dt": dt,
        "target_deg": float(target_deg),
        "v_ref": float(v_ref),
        "roll0_deg": float(roll0_deg),
        "controller": np.array([controller]),
        "final_reason": np.array([final_reason]),
        **{k: np.array([v]) for k, v in metrics.items()},
    }
    if horizon is not None:
        payload["horizon"] = np.array([horizon])
    if candidates is not None:
        payload["candidates"] = np.array([candidates])
    if uncertainty_weight is not None:
        payload["uncertainty_weight"] = np.array([uncertainty_weight])
    if best_costs is not None:
        payload["best_costs"] = best_costs
    if uncertainty_costs is not None:
        payload["uncertainty_costs"] = uncertainty_costs
    np.savez(out, **payload)
    return metrics
