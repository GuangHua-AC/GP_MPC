from __future__ import annotations

from pathlib import Path

import numpy as np


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def controller_label(kind: str) -> str:
    if kind == "pd":
        return "PD/VMC-terrain"
    if kind == "nn_mpc":
        return "NN-MPC"
    if kind == "gp_mpc":
        return "GP-MPC"
    return kind


def terrain_filename(kind: str, terrain_mode: str, v_ref: float, backend: str | None = None) -> str:
    backend_part = f"_{backend}" if backend else ""
    return f"terrain_{terrain_mode}_v{tag_float(v_ref)}_{kind}{backend_part}.npz"


def compute_metrics(
    states: np.ndarray,
    actions: np.ndarray,
    terrain_diffs: np.ndarray,
    support_rolls: np.ndarray,
    final_reason: str,
) -> dict[str, float | int | str | bool]:
    if len(states) == 0:
        return {
            "steps": 0,
            "final_reason": final_reason,
            "success": False,
            "max_abs_theta_rad": 0.0,
            "max_abs_phi_rad": 0.0,
            "max_abs_roll_rad": 0.0,
            "max_abs_support_roll_rad": 0.0,
            "max_abs_terrain_diff_m": 0.0,
            "max_abs_leg_diff_m": 0.0,
            "max_abs_x_m": 0.0,
            "action_energy": 0.0,
        }
    max_theta = float(np.max(np.abs(states[:, 0])))
    max_phi = float(np.max(np.abs(states[:, 4])))
    max_roll = float(np.max(np.abs(states[:, 8])))
    max_support_roll = float(np.max(np.abs(support_rolls))) if len(support_rolls) else 0.0
    max_terrain_diff = float(np.max(np.abs(terrain_diffs))) if len(terrain_diffs) else 0.0
    max_leg_diff = float(np.max(np.abs(states[:, 12])))
    max_x = float(np.max(np.abs(states[:, 2])))
    action_energy = float(np.sum(actions**2)) if len(actions) else 0.0
    success = (
        final_reason in {"not_done", "max_steps"}
        and max_theta < 0.8
        and max_phi < 0.8
        and max_roll < 0.20
        and max_support_roll < 0.20
    )
    return {
        "steps": int(len(states)),
        "final_reason": final_reason,
        "success": bool(success),
        "max_abs_theta_rad": max_theta,
        "max_abs_phi_rad": max_phi,
        "max_abs_roll_rad": max_roll,
        "max_abs_support_roll_rad": max_support_roll,
        "max_abs_terrain_diff_m": max_terrain_diff,
        "max_abs_leg_diff_m": max_leg_diff,
        "max_abs_x_m": max_x,
        "action_energy": action_energy,
    }


def save_terrain_result(
    out: Path,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    left_heights: np.ndarray,
    right_heights: np.ndarray,
    terrain_diffs: np.ndarray,
    support_rolls: np.ndarray,
    dt: float,
    terrain_mode: str,
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
    metrics = compute_metrics(states, actions, terrain_diffs, support_rolls, final_reason)
    payload = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "left_heights": left_heights,
        "right_heights": right_heights,
        "terrain_diffs": terrain_diffs,
        "support_rolls": support_rolls,
        "dt": np.array([dt]),
        "terrain_mode": np.array([terrain_mode]),
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
