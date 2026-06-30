from __future__ import annotations

import numpy as np

from wheel_legged.dynamics.terrain import terrain_heights


def _angle_error(angle: float, ref: float) -> float:
    return float((angle - ref + np.pi) % (2.0 * np.pi) - np.pi)


def safe_norm_std(std) -> np.ndarray:
    arr = np.asarray(std, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=0.0)
    return np.linalg.norm(np.maximum(arr, 0.0), axis=1)


def terminal_state_cost(state, env, ref) -> float:
    s = np.asarray(state, dtype=float).reshape(-1)
    theta = s[0]
    x = s[2]
    x_dot = s[3]
    phi = s[4]
    yaw = s[6]
    yaw_dot = s[7]
    roll = s[8]
    roll_dot = s[9]
    alpha = s[10]
    alpha_dot = s[11]
    cost = (
        40.0 * theta**2
        + 40.0 * phi**2
        + 2.0 * (x - ref.x_ref) ** 2
        + 4.0 * (x_dot - ref.v_ref) ** 2
    )
    if getattr(env, "task", "") in {"balance_turn", "balance_turn_roll"}:
        yaw_err = _angle_error(float(yaw), float(ref.yaw_ref))
        roll_err = float(roll - ref.roll_ref)
        cost += (
            80.0 * yaw_err**2
            + 6.0 * yaw_dot**2
            + 120.0 * roll_err**2
            + 8.0 * roll_dot**2
        )
    if getattr(env, "task", "") == "height":
        l0 = env.leg.L0(float(alpha))
        l0_dot = env.leg.L0_dot(float(alpha), float(alpha_dot))
        cost += 180.0 * (l0 - ref.L0_ref) ** 2 + 8.0 * (l0_dot - ref.L0_dot_ref) ** 2
    if getattr(env, "task", "") == "terrain":
        leg_diff = s[12]
        leg_diff_dot = s[13]
        left_h, right_h = terrain_heights(float(x), env.terrain_mode, env.p)
        terrain_diff = left_h - right_h
        leg_diff_ref = -terrain_diff
        support_roll = np.arctan2(terrain_diff + leg_diff, env.p.D)
        cost += (
            80.0 * roll**2
            + 6.0 * roll_dot**2
            + 80.0 * support_roll**2
            + 120.0 * (leg_diff - leg_diff_ref) ** 2
            + 4.0 * leg_diff_dot**2
        )
    return float(np.nan_to_num(cost, nan=1e8, posinf=1e8, neginf=1e8))


def summarize_rollout_metrics(states, actions, infos=None) -> dict[str, float | int | str]:
    states_np = np.asarray(states, dtype=float)
    actions_np = np.asarray(actions, dtype=float)
    if states_np.size == 0:
        return {
            "steps": 0,
            "max_abs_theta": 0.0,
            "max_abs_phi": 0.0,
            "max_abs_x": 0.0,
            "final_x": 0.0,
            "action_norm_mean": 0.0,
            "final_reason": "no_steps",
        }

    final_reason = "not_done"
    if infos:
        final_reason = str(infos[-1].get("final_reason", "not_done"))

    return {
        "steps": int(len(states_np)),
        "max_abs_theta": float(np.max(np.abs(states_np[:, 0]))),
        "max_abs_phi": float(np.max(np.abs(states_np[:, 4]))),
        "max_abs_x": float(np.max(np.abs(states_np[:, 2]))),
        "final_x": float(states_np[-1, 2]),
        "action_norm_mean": float(np.mean(np.linalg.norm(actions_np, axis=1))) if len(actions_np) else 0.0,
        "final_reason": final_reason,
    }
