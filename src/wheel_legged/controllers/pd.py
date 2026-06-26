from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wheel_legged.dynamics.env import Reference, WheelLeggedEnv
from wheel_legged.dynamics.pdf_model import equivalent_height


@dataclass
class PDGains:
    kp_theta: float = 3.00
    kd_theta: float = 0.55
    kp_x: float = 0.30
    kd_x: float = 0.40
    kp_phi: float = 8.00
    kd_phi: float = 1.20

    kp_yaw: float = 0.18
    kd_yaw: float = 0.06

    kp_roll: float = 5.0
    kd_roll: float = 1.2
    roll_gravity_ff_scale: float = 1.0
    roll_centrifugal_ff_scale: float = 0.0

    kp_L0: float = 650.0
    kd_L0: float = 45.0

    kp_leg_diff_roll: float = 0.08
    kd_leg_diff_roll: float = 0.02


class WheelLeggedPDController:
    """PD controller matching the six action channels of WheelLeggedEnv."""

    def __init__(self, env: WheelLeggedEnv, gains: PDGains | None = None):
        self.env = env
        self.g = gains or PDGains()

    def act(self, state: np.ndarray, ref: Reference | None = None) -> np.ndarray:
        ref = ref or self.env.ref
        p = self.env.p
        g = self.g
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
            _leg_diff_dot,
        ) = np.asarray(state, dtype=float)

        T = (
            g.kp_theta * theta
            + g.kd_theta * theta_dot
            + g.kd_x * (x_dot - ref.v_ref)
        )

        Tp = -g.kp_phi * phi - g.kd_phi * phi_dot
        Tyaw = g.kp_yaw * (ref.yaw_ref - yaw) + g.kd_yaw * (0.0 - yaw_dot)

        h = equivalent_height(theta, phi, p)
        mass_roll = p.M + 2.0 * p.mp
        gravity_torque = mass_roll * p.g * h * np.sin(roll)
        centrifugal_torque = mass_roll * x_dot * yaw_dot * h * np.cos(roll)
        desired_roll_torque = (
            -g.kp_roll * (roll - ref.roll_ref)
            -g.kd_roll * roll_dot
            -g.roll_gravity_ff_scale * gravity_torque
            -g.roll_centrifugal_ff_scale * centrifugal_torque
        )
        Froll = 2.0 * desired_roll_torque / p.D

        L0 = self.env.leg.L0(alpha)
        L0_dot = self.env.leg.L0_dot(alpha, alpha_dot)
        Fff = (p.M + p.mp * p.eta) * p.g
        Fheight = Fff + g.kp_L0 * (ref.L0_ref - L0) + g.kd_L0 * (ref.L0_dot_ref - L0_dot)

        info = self.env.info(state, "not_done")
        desired_leg_diff = -info["terrain_diff"]
        leg_diff_cmd = desired_leg_diff - g.kp_leg_diff_roll * roll - g.kd_leg_diff_roll * roll_dot

        action = np.array([T, Tp, Tyaw, Froll, Fheight, leg_diff_cmd], dtype=float)
        return self.env._clip_action(action)
