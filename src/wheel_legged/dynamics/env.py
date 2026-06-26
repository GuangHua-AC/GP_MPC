from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .parameters import WheelLeggedParams
from .pdf_model import (
    VirtualLegKinematics,
    height_alpha_ddot,
    roll_ddot,
    solve_planar_accelerations,
    yaw_ddot_from_torque,
)
from .terrain import terrain_heights


TASKS = {"balance", "balance_turn", "balance_turn_roll", "height", "terrain"}


@dataclass
class Reference:
    x_ref: float = 0.0
    x_limit: float = 5.0
    v_ref: float = 0.0
    yaw_ref: float = 0.0
    roll_ref: float = 0.0
    L0_ref: float = 0.32
    L0_dot_ref: float = 0.0


class WheelLeggedEnv:
    """
    Unified simulation environment.

    State:
        [theta, theta_dot, x, x_dot, phi, phi_dot,
         delta, delta_dot, psi, psi_dot, alpha, alpha_dot,
         leg_diff, leg_diff_dot]

    Action:
        [T, Tp, Tyaw, Froll, Fheight, leg_diff_cmd]
    """

    state_dim = 14
    action_dim = 6

    def __init__(
        self,
        task: str = "balance_turn_roll",
        params: WheelLeggedParams | None = None,
        terrain_mode: str = "left_obstacle",
    ):
        if task not in TASKS:
            raise ValueError(f"unknown task {task}; choose from {sorted(TASKS)}")
        self.task = task
        self.p = params or WheelLeggedParams()
        self.leg = VirtualLegKinematics(self.p)
        self.terrain_mode = terrain_mode if task == "terrain" else "flat"
        self.state = np.zeros(self.state_dim, dtype=float)
        self.step_count = 0
        self.final_reason = "not_done"
        self.ref = Reference()
        self.external_force_x = 0.0

    def set_external_force_x(self, force: float) -> None:
        """Set horizontal body disturbance force in Newtons."""
        self.external_force_x = float(force)

    def reset(
        self,
        theta0: float = 0.03,
        phi0: float = 0.03,
        x0: float = 0.0,
        yaw0: float = 0.0,
        roll0: float = 0.0,
        L0_init: float = 0.32,
        ref: Reference | None = None,
    ) -> np.ndarray:
        self.ref = ref or Reference()
        alpha0 = self.leg.alpha_from_L0(L0_init)
        self.state = np.array(
            [
                theta0,
                0.0,
                x0,
                self.ref.v_ref,
                phi0,
                0.0,
                yaw0,
                0.0,
                roll0,
                0.0,
                alpha0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        self.step_count = 0
        self.final_reason = "not_done"
        self.external_force_x = 0.0
        return self.state.copy()

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        p = self.p
        a = np.asarray(action, dtype=float).reshape(self.action_dim).copy()
        a[0] = np.clip(a[0], -p.T_limit, p.T_limit)
        a[1] = np.clip(a[1], -p.Tp_limit, p.Tp_limit)
        a[2] = np.clip(a[2], -p.Tyaw_limit, p.Tyaw_limit)
        a[3] = np.clip(a[3], -p.Froll_limit, p.Froll_limit)
        a[4] = np.clip(a[4], p.Fheight_min, p.Fheight_max)
        a[5] = np.clip(a[5], -p.leg_diff_cmd_limit, p.leg_diff_cmd_limit)

        if self.task == "balance":
            a[2:] = 0.0
        elif self.task == "balance_turn":
            a[3:] = 0.0
        elif self.task == "balance_turn_roll":
            a[4:] = 0.0
        elif self.task == "height":
            a[2] = 0.0
            a[3] = 0.0
            a[5] = 0.0
        return a

    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        p = self.p
        s = np.asarray(state, dtype=float)
        a = self._clip_action(action)
        (
            theta,
            theta_dot,
            x,
            x_dot,
            phi,
            phi_dot,
            yaw,
            yaw_dot,
            roll,
            roll_dot,
            alpha,
            alpha_dot,
            leg_diff,
            leg_diff_dot,
        ) = s

        theta_ddot, x_ddot, phi_ddot = solve_planar_accelerations(
            s[:6],
            a[:2],
            p,
            external_force_x=self.external_force_x,
        )
        yaw_ddot = yaw_ddot_from_torque(a[2], p)

        Froll = a[3]
        if self.task == "terrain":
            left_h, right_h = terrain_heights(x, self.terrain_mode, p)
            terrain_diff = left_h - right_h
            support_roll = np.arctan2(terrain_diff + leg_diff, p.D)
            Froll += 30.0 * (support_roll - roll)

        if self.task in {"balance_turn_roll", "terrain"}:
            roll_state_dot = roll_dot
            psi_ddot = roll_ddot(theta, phi, x_dot, yaw_dot, roll, roll_dot, Froll, p)
        else:
            roll_state_dot = 0.0
            psi_ddot = 0.0
        alpha = self.leg.clip_alpha(alpha)
        if self.task in {"height", "terrain"}:
            alpha_state_dot = alpha_dot
            alpha_ddot = height_alpha_ddot(alpha, alpha_dot, a[4], self.leg, p)
        else:
            alpha_state_dot = 0.0
            alpha_ddot = 0.0
        leg_diff = float(np.clip(leg_diff, -p.leg_diff_limit, p.leg_diff_limit))
        if self.task == "terrain":
            leg_diff_state_dot = leg_diff_dot
            leg_diff_ddot = (
                p.leg_diff_omega**2 * (a[5] - leg_diff)
                - 2.0 * p.leg_diff_zeta * p.leg_diff_omega * leg_diff_dot
            )
        else:
            leg_diff_state_dot = 0.0
            leg_diff_ddot = 0.0

        deriv = np.array(
            [
                theta_dot,
                theta_ddot,
                x_dot,
                x_ddot,
                phi_dot,
                phi_ddot,
                yaw_dot,
                yaw_ddot,
                roll_state_dot,
                psi_ddot,
                alpha_state_dot,
                alpha_ddot,
                leg_diff_state_dot,
                leg_diff_ddot,
            ],
            dtype=float,
        )
        deriv = np.nan_to_num(deriv, nan=0.0, posinf=1e3, neginf=-1e3)
        return np.clip(deriv, -1e3, 1e3)

    def step(self, action: np.ndarray, ref: Reference | None = None) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if ref is not None:
            self.ref = ref

        a = self._clip_action(action)
        s = self.state
        dt = self.p.dt
        k1 = self.dynamics(s, a)
        k2 = self.dynamics(s + 0.5 * dt * k1, a)
        k3 = self.dynamics(s + 0.5 * dt * k2, a)
        k4 = self.dynamics(s + dt * k3, a)
        ns = s + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        ns = np.nan_to_num(ns, nan=0.0, posinf=1e3, neginf=-1e3)
        ns[10] = self.leg.clip_alpha(ns[10])
        ns[12] = np.clip(ns[12], -self.p.leg_diff_limit, self.p.leg_diff_limit)
        self.state = ns
        self.step_count += 1
        reward = -self.cost(ns, a, self.ref)
        done, reason = self._done(ns)
        self.final_reason = reason
        return ns.copy(), float(reward), done, self.info(ns, reason)

    def cost(self, state: np.ndarray, action: np.ndarray, ref: Reference | None = None) -> float:
        ref = ref or self.ref
        s = np.asarray(state, dtype=float)
        a = np.asarray(action, dtype=float)
        theta, theta_dot, x, x_dot, phi, phi_dot, yaw, yaw_dot, roll, roll_dot, alpha, alpha_dot, leg_diff, _ = s
        L0 = self.leg.L0(alpha)
        L0_dot = self.leg.L0_dot(alpha, alpha_dot)
        left_h, right_h = terrain_heights(x, self.terrain_mode, self.p)
        terrain_diff = left_h - right_h
        support_roll = np.arctan2(terrain_diff + leg_diff, self.p.D)

        cost = (
            30.0 * theta**2
            + 2.0 * theta_dot**2
            + 1.0 * (x_dot - ref.v_ref) ** 2
            + 30.0 * phi**2
            + 2.0 * phi_dot**2
            + 35.0 * (yaw - ref.yaw_ref) ** 2
            + 2.0 * yaw_dot**2
            + 100.0 * (roll - ref.roll_ref) ** 2
            + 6.0 * roll_dot**2
            + 100.0 * (L0 - ref.L0_ref) ** 2
            + 4.0 * (L0_dot - ref.L0_dot_ref) ** 2
            + 50.0 * support_roll**2
            + 0.01 * float(np.sum(a[:5] ** 2))
            + 0.5 * float(a[5] ** 2)
        )
        return float(np.clip(cost, 0.0, 1e8))

    def _done(self, state: np.ndarray) -> tuple[bool, str]:
        p = self.p
        theta, _theta_dot, x, _x_dot, phi, _phi_dot, yaw, _yaw_dot, roll, _roll_dot = state[:10]
        if not np.all(np.isfinite(state)):
            return True, "nan_or_inf"
        if abs(theta) > p.max_abs_theta:
            return True, "fall_theta"
        if abs(phi) > p.max_abs_phi:
            return True, "fall_phi"
        if self.task in {"balance_turn_roll", "terrain"} and abs(roll) > p.max_abs_roll:
            return True, "fall_roll"
        x_limit = self.ref.x_limit if self.ref.x_limit is not None else p.max_abs_x
        if abs(x - self.ref.x_ref) > x_limit:
            return True, "x_out"
        if abs(yaw) > p.max_abs_yaw:
            return True, "yaw_out"
        if self.step_count >= p.max_steps:
            return True, "max_steps"
        return False, "not_done"

    def info(self, state: np.ndarray, reason: str) -> dict[str, Any]:
        x = float(state[2])
        alpha = float(state[10])
        alpha_dot = float(state[11])
        left_h, right_h = terrain_heights(x, self.terrain_mode, self.p)
        terrain_diff = left_h - right_h
        return {
            "final_reason": reason,
            "L0": self.leg.L0(alpha),
            "L0_dot": self.leg.L0_dot(alpha, alpha_dot),
            "left_ground_height": float(left_h),
            "right_ground_height": float(right_h),
            "terrain_diff": float(terrain_diff),
            "support_roll": float(np.arctan2(terrain_diff + state[12], self.p.D)),
            "external_force_x": float(self.external_force_x),
        }
