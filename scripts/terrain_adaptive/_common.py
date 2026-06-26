from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wheel_legged.controllers.pd import PDGains
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv
from wheel_legged.dynamics.pdf_model import equivalent_height
from wheel_legged.dynamics.terrain import terrain_heights
from wheel_legged.utils.paths import OUTPUT_DIR


ADAPTIVE_DIR = OUTPUT_DIR / "terrain_adaptive"


def ensure_adaptive_dirs() -> None:
    for name in ["pd", "mpc", "models", "metrics", "videos", "final"]:
        (ADAPTIVE_DIR / name).mkdir(parents=True, exist_ok=True)


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def adaptive_filename(kind: str, terrain_mode: str, v_ref: float, backend: str | None = None) -> str:
    backend_part = f"_{backend}" if backend else ""
    return f"terrain_adaptive_{terrain_mode}_v{tag_float(v_ref)}_{kind}{backend_part}.npz"


def controller_label(kind: str) -> str:
    labels = {
        "adaptive_pd": "Adaptive-PD/VMC",
        "adaptive_nn_mpc": "Adaptive-NN-MPC",
        "adaptive_gp_mpc": "Adaptive-GP-MPC",
    }
    return labels.get(kind, kind)


@dataclass
class AdaptiveGains:
    adapt_rate: float = 4.0
    adapt_damping: float = 0.10
    leak: float = 0.02
    deadband_deg: float = 0.15


class OnlineTerrainAdaptiveController:
    """Blind terrain adaptation.

    This controller never reads the terrain height difference from the
    simulator. It estimates the needed left-right leg height difference from
    measured body roll and roll rate after the robot has contacted the terrain.
    """

    def __init__(
        self,
        env: WheelLeggedEnv,
        gains: PDGains | None = None,
        adaptive_gains: AdaptiveGains | None = None,
    ):
        self.env = env
        self.g = gains or PDGains()
        self.ag = adaptive_gains or AdaptiveGains()
        self.leg_diff_ref = 0.0
        self.last_action: np.ndarray | None = None
        self.last_leg_diff_ref = 0.0

    def reset(self) -> None:
        self.leg_diff_ref = 0.0
        self.last_action = None
        self.last_leg_diff_ref = 0.0

    def _update_leg_diff_ref(self, roll: float, roll_dot: float) -> None:
        p = self.env.p
        deadband = np.deg2rad(self.ag.deadband_deg)
        if abs(roll) > deadband or abs(roll_dot) > 0.01:
            self.leg_diff_ref += (-self.ag.adapt_rate * roll - self.ag.adapt_damping * roll_dot) * p.dt
        else:
            self.leg_diff_ref *= 1.0 - self.ag.leak * p.dt
        self.leg_diff_ref = float(np.clip(self.leg_diff_ref, -p.leg_diff_limit, p.leg_diff_limit))
        self.last_leg_diff_ref = self.leg_diff_ref

    def act(self, state: np.ndarray, ref: Reference | None = None) -> np.ndarray:
        ref = ref or self.env.ref
        p = self.env.p
        g = self.g
        (
            theta,
            theta_dot,
            _x,
            x_dot,
            phi,
            phi_dot,
            yaw,
            yaw_dot,
            roll,
            roll_dot,
            alpha,
            alpha_dot,
            _leg_diff,
            _leg_diff_dot,
        ) = np.asarray(state, dtype=float)

        self._update_leg_diff_ref(roll, roll_dot)

        T = g.kp_theta * theta + g.kd_theta * theta_dot + g.kd_x * (x_dot - ref.v_ref)
        Tp = -g.kp_phi * phi - g.kd_phi * phi_dot
        Tyaw = g.kp_yaw * (ref.yaw_ref - yaw) + g.kd_yaw * (0.0 - yaw_dot)

        h = equivalent_height(theta, phi, p)
        mass_roll = p.M + 2.0 * p.mp
        gravity_torque = mass_roll * p.g * h * np.sin(roll)
        centrifugal_torque = mass_roll * x_dot * yaw_dot * h * np.cos(roll)
        desired_roll_torque = (
            -g.kp_roll * (roll - ref.roll_ref)
            - g.kd_roll * roll_dot
            - g.roll_gravity_ff_scale * gravity_torque
            - g.roll_centrifugal_ff_scale * centrifugal_torque
        )
        Froll = 2.0 * desired_roll_torque / p.D

        L0 = self.env.leg.L0(alpha)
        L0_dot = self.env.leg.L0_dot(alpha, alpha_dot)
        Fff = (p.M + p.mp * p.eta) * p.g
        Fheight = Fff + g.kp_L0 * (ref.L0_ref - L0) + g.kd_L0 * (ref.L0_dot_ref - L0_dot)

        leg_diff_cmd = self.leg_diff_ref - g.kp_leg_diff_roll * roll - g.kd_leg_diff_roll * roll_dot
        action = self.env._clip_action(np.array([T, Tp, Tyaw, Froll, Fheight, leg_diff_cmd], dtype=float))
        self.last_action = action.copy()
        return action


class BlindTerrainCostEnv(WheelLeggedEnv):
    """Planning cost for unknown terrain.

    The real simulator still contains terrain, but the MPC cost intentionally
    does not use terrain_heights or terrain_diff. It only sees the predicted
    robot state and penalizes unstable attitude.
    """

    def cost(self, state: np.ndarray, action: np.ndarray, ref: Reference | None = None) -> float:
        ref = ref or self.ref
        s = np.asarray(state, dtype=float)
        a = np.asarray(action, dtype=float)
        theta, theta_dot, x, x_dot, phi, phi_dot, yaw, yaw_dot, roll, roll_dot, alpha, alpha_dot, leg_diff, leg_diff_dot = s
        L0 = self.leg.L0(alpha)
        L0_dot = self.leg.L0_dot(alpha, alpha_dot)
        x_margin = max(0.0, abs(x - ref.x_ref) - ref.x_limit)
        cost = (
            35.0 * theta**2
            + 2.5 * theta_dot**2
            + 1.0 * (x_dot - ref.v_ref) ** 2
            + 35.0 * phi**2
            + 2.5 * phi_dot**2
            + 20.0 * (yaw - ref.yaw_ref) ** 2
            + 1.5 * yaw_dot**2
            + 180.0 * (roll - ref.roll_ref) ** 2
            + 8.0 * roll_dot**2
            + 80.0 * (L0 - ref.L0_ref) ** 2
            + 3.0 * (L0_dot - ref.L0_dot_ref) ** 2
            + 2.0 * leg_diff**2
            + 0.5 * leg_diff_dot**2
            + 1000.0 * x_margin**2
            + 0.01 * float(np.sum(a[:5] ** 2))
            + 0.3 * float(a[5] ** 2)
        )
        return float(np.clip(cost, 0.0, 1e8))


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
            "terrain_comp_rmse_m": 0.0,
            "max_abs_x_m": 0.0,
            "action_energy": 0.0,
        }
    max_theta = float(np.max(np.abs(states[:, 0])))
    max_phi = float(np.max(np.abs(states[:, 4])))
    max_roll = float(np.max(np.abs(states[:, 8])))
    max_support_roll = float(np.max(np.abs(support_rolls))) if len(support_rolls) else 0.0
    max_terrain_diff = float(np.max(np.abs(terrain_diffs))) if len(terrain_diffs) else 0.0
    max_leg_diff = float(np.max(np.abs(states[:, 12])))
    terrain_comp_rmse = float(np.sqrt(np.mean((states[:, 12] + terrain_diffs) ** 2))) if len(terrain_diffs) else 0.0
    max_x = float(np.max(np.abs(states[:, 2])))
    action_energy = float(np.sum(actions**2)) if len(actions) else 0.0
    success = (
        final_reason in {"not_done", "max_steps"}
        and max_theta < 0.8
        and max_phi < 0.8
        and max_roll < 0.25
        and max_support_roll < 0.25
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
        "terrain_comp_rmse_m": terrain_comp_rmse,
        "max_abs_x_m": max_x,
        "action_energy": action_energy,
    }


def save_adaptive_result(
    out: Path,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    left_heights: np.ndarray,
    right_heights: np.ndarray,
    terrain_diffs: np.ndarray,
    support_rolls: np.ndarray,
    leg_diff_refs: np.ndarray,
    dt: float,
    terrain_mode: str,
    v_ref: float,
    controller: str,
    final_reason: str,
    horizon: int | None = None,
    candidates: int | None = None,
    uncertainty_weight: float | None = None,
    mpc_blend: float | None = None,
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
        "leg_diff_refs": leg_diff_refs,
        "dt": np.array([dt]),
        "terrain_mode": np.array([terrain_mode]),
        "v_ref": np.array([v_ref]),
        "controller": np.array([controller]),
        "final_reason": np.array([final_reason]),
        "terrain_known_to_controller": np.array([False]),
        **{k: np.array([v]) for k, v in metrics.items()},
    }
    if horizon is not None:
        payload["horizon"] = np.array([horizon])
    if candidates is not None:
        payload["candidates"] = np.array([candidates])
    if uncertainty_weight is not None:
        payload["uncertainty_weight"] = np.array([uncertainty_weight])
    if mpc_blend is not None:
        payload["mpc_blend"] = np.array([mpc_blend])
    if best_costs is not None:
        payload["best_costs"] = best_costs
    if uncertainty_costs is not None:
        payload["uncertainty_costs"] = uncertainty_costs
    np.savez(out, **payload)
    return metrics


def terrain_trace_for_state(env: WheelLeggedEnv, state: np.ndarray) -> tuple[float, float, float, float]:
    left_h, right_h = terrain_heights(float(state[2]), env.terrain_mode, env.p)
    terrain_diff = float(left_h - right_h)
    support_roll = float(np.arctan2(terrain_diff + state[12], env.p.D))
    return float(left_h), float(right_h), terrain_diff, support_roll
