from __future__ import annotations

from pathlib import Path

import numpy as np


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def controller_label(kind: str) -> str:
    if kind == "pd":
        return "PD/VMC-height"
    if kind == "nn_mpc":
        return "NN-MPC"
    if kind == "gp_mpc":
        return "GP-MPC"
    return kind


def height_filename(kind: str, mode: str, low: float, high: float, v_ref: float, backend: str | None = None) -> str:
    backend_part = f"_{backend}" if backend else ""
    return f"height_{mode}_L{tag_float(low)}_to_{tag_float(high)}_v{tag_float(v_ref)}_{kind}{backend_part}.npz"


def initial_l0(mode: str, low: float, high: float, mid: float) -> float:
    if mode in {"step", "step_cycle"}:
        return low
    if mode == "down":
        return high
    return mid


def l0_reference(
    mode: str,
    step: int,
    dt: float,
    *,
    low: float,
    high: float,
    switch_time: float,
    sine_freq: float,
) -> tuple[float, float]:
    t = step * dt
    mid = 0.5 * (low + high)
    amp = 0.5 * (high - low)
    if mode == "fixed":
        return mid, 0.0
    if mode == "step":
        return (low if t < switch_time else high), 0.0
    if mode == "step_cycle":
        if t < switch_time:
            return low, 0.0
        if t < 2.0 * switch_time:
            return high, 0.0
        return low, 0.0
    if mode == "down":
        return (high if t < switch_time else low), 0.0
    if mode == "sine":
        omega = 2.0 * np.pi * sine_freq
        return mid + amp * np.sin(omega * t), amp * omega * np.cos(omega * t)
    raise ValueError(f"unknown height mode: {mode}")


def compute_l0s(states: np.ndarray, offset: float = 0.30, gain: float = 0.10) -> np.ndarray:
    if len(states) == 0:
        return np.zeros(0, dtype=float)
    return offset + gain * np.sin(states[:, 10])


def compute_metrics(
    states: np.ndarray,
    actions: np.ndarray,
    l0s: np.ndarray,
    l0_refs: np.ndarray,
    final_reason: str,
) -> dict[str, float | int | str | bool]:
    if len(states) == 0:
        return {
            "steps": 0,
            "final_reason": final_reason,
            "success": False,
            "l0_rmse_m": 0.0,
            "max_abs_l0_error_m": 0.0,
            "final_l0_error_m": 0.0,
            "max_abs_theta_rad": 0.0,
            "max_abs_phi_rad": 0.0,
            "max_abs_x_m": 0.0,
            "action_energy": 0.0,
        }

    n = min(len(states), len(l0s), len(l0_refs))
    l0_err = l0s[:n] - l0_refs[:n]
    l0_rmse = float(np.sqrt(np.mean(l0_err**2))) if n else 0.0
    max_l0_err = float(np.max(np.abs(l0_err))) if n else 0.0
    final_l0_err = float(l0_err[-1]) if n else 0.0
    max_theta = float(np.max(np.abs(states[:, 0])))
    max_phi = float(np.max(np.abs(states[:, 4])))
    max_x = float(np.max(np.abs(states[:, 2])))
    action_energy = float(np.sum(actions**2)) if len(actions) else 0.0
    success = (
        final_reason in {"not_done", "max_steps"}
        and abs(final_l0_err) < 0.015
        and max_theta < 0.8
        and max_phi < 0.8
    )
    return {
        "steps": int(len(states)),
        "final_reason": final_reason,
        "success": bool(success),
        "l0_rmse_m": l0_rmse,
        "max_abs_l0_error_m": max_l0_err,
        "final_l0_error_m": final_l0_err,
        "max_abs_theta_rad": max_theta,
        "max_abs_phi_rad": max_phi,
        "max_abs_x_m": max_x,
        "action_energy": action_energy,
    }


def save_height_result(
    out: Path,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    l0s: np.ndarray,
    l0_refs: np.ndarray,
    l0_dot_refs: np.ndarray,
    dt: float,
    mode: str,
    low: float,
    high: float,
    v_ref: float,
    controller: str,
    final_reason: str,
    horizon: int | None = None,
    candidates: int | None = None,
    uncertainty_weight: float | None = None,
    best_costs: np.ndarray | None = None,
    uncertainty_costs: np.ndarray | None = None,
) -> dict[str, float | int | str | bool]:
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(states, actions, l0s, l0_refs, final_reason)
    payload = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "L0s": l0s,
        "L0_refs": l0_refs,
        "L0_dot_refs": l0_dot_refs,
        "dt": np.array([dt]),
        "mode": np.array([mode]),
        "low": np.array([low]),
        "high": np.array([high]),
        "v_ref": np.array([v_ref]),
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
