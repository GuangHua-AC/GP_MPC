from __future__ import annotations

import numpy as np

from .parameters import WheelLeggedParams


class VirtualLegKinematics:
    """Replace this class when the real five-bar geometry is available."""

    def __init__(self, params: WheelLeggedParams):
        self.p = params

    def clip_alpha(self, alpha: float) -> float:
        return float(np.clip(alpha, self.p.alpha_min, self.p.alpha_max))

    def L0(self, alpha: float) -> float:
        alpha = self.clip_alpha(alpha)
        return float(self.p.L0_offset + self.p.L0_gain * np.sin(alpha))

    def J1(self, alpha: float) -> float:
        alpha = self.clip_alpha(alpha)
        return float(self.p.L0_gain * np.cos(alpha))

    def J2(self, alpha: float) -> float:
        alpha = self.clip_alpha(alpha)
        return float(-self.p.L0_gain * np.sin(alpha))

    def alpha_from_L0(self, L0: float) -> float:
        z = np.clip((L0 - self.p.L0_offset) / self.p.L0_gain, -0.95, 0.95)
        return self.clip_alpha(float(np.arcsin(z)))

    def L0_dot(self, alpha: float, alpha_dot: float) -> float:
        return float(self.J1(alpha) * alpha_dot)


def planar_intermediate_forces(
    theta: float,
    theta_dot: float,
    phi: float,
    phi_dot: float,
    theta_ddot: float,
    x_ddot: float,
    phi_ddot: float,
    p: WheelLeggedParams,
    external_force_x: float = 0.0,
) -> tuple[float, float, float, float]:
    s_th, c_th = np.sin(theta), np.cos(theta)
    s_ph, c_ph = np.sin(phi), np.cos(phi)

    Nm = p.M * (
        x_ddot
        + (p.L + p.Lm) * theta_ddot * c_th
        - (p.L + p.Lm) * theta_dot**2 * s_th
        - p.l * phi_ddot * c_ph
        + p.l * phi_dot**2 * s_ph
    ) - external_force_x

    Pm = p.M * p.g - p.M * (
        (p.L + p.Lm) * (theta_ddot * s_th + theta_dot**2 * c_th)
        + p.l * (phi_ddot * s_ph + phi_dot**2 * c_ph)
    )

    N = (
        (p.M + p.mp) * (x_ddot + p.L * theta_ddot * c_th - p.L * theta_dot**2 * s_th)
        + p.M * p.Lm * theta_ddot * c_th
        - p.M * p.Lm * theta_dot**2 * s_th
        - p.M * p.l * phi_ddot * c_ph
        + p.M * p.l * phi_dot**2 * s_ph
        - external_force_x
    )

    P = (
        (p.M + p.mp) * p.g
        - ((p.M + p.mp) * p.L + p.M * p.Lm) * theta_ddot * s_th
        - ((p.M + p.mp) * p.L + p.M * p.Lm) * theta_dot**2 * c_th
        - p.M * p.l * phi_ddot * s_ph
        - p.M * p.l * phi_dot**2 * c_ph
    )

    return float(N), float(P), float(Nm), float(Pm)


def solve_planar_accelerations(
    state6: np.ndarray,
    action2: np.ndarray,
    p: WheelLeggedParams,
    external_force_x: float = 0.0,
) -> np.ndarray:
    """Solve PDF equations (1), (7), and (14) for [theta_ddot, x_ddot, phi_ddot].

    external_force_x is a horizontal force applied to the body COM. Positive
    force pushes the robot in +x direction.
    """

    theta, theta_dot, _x, _x_dot, phi, phi_dot = np.asarray(state6, dtype=float)
    T, Tp = np.asarray(action2, dtype=float)

    def residual(q: np.ndarray) -> np.ndarray:
        theta_ddot, x_ddot, phi_ddot = q
        N, P, Nm, Pm = planar_intermediate_forces(
            theta,
            theta_dot,
            phi,
            phi_dot,
            theta_ddot,
            x_ddot,
            phi_ddot,
            p,
            external_force_x,
        )
        wheel_den = p.Iw / p.R + p.mw * p.R
        r_x = x_ddot - (T - N * p.R) / wheel_den
        r_theta = theta_ddot - (
            (P * p.L + Pm * p.Lm) * np.sin(theta)
            - (N * p.L + Nm * p.Lm) * np.cos(theta)
            - T
            + Tp
        ) / p.Ip
        r_phi = phi_ddot - (Tp + Nm * p.l * np.cos(phi) + Pm * p.l * np.sin(phi)) / p.Im
        return np.array([r_theta, r_x, r_phi], dtype=float)

    r0 = residual(np.zeros(3))
    cols = []
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        cols.append(residual(e) - r0)
    a_mat = np.column_stack(cols)

    try:
        q = np.linalg.solve(a_mat, -r0)
    except np.linalg.LinAlgError:
        q = np.linalg.lstsq(a_mat, -r0, rcond=None)[0]

    return np.nan_to_num(q, nan=0.0, posinf=1e3, neginf=-1e3)


def yaw_ddot_from_torque(Tyaw: float, p: WheelLeggedParams) -> float:
    denom = p.R * ((p.mw + p.Iw / (p.R**2)) * p.D + 2.0 * p.Iz / p.D)
    return float(Tyaw / denom)


def equivalent_height(theta: float, phi: float, p: WheelLeggedParams) -> float:
    denom = p.M + 2.0 * p.mp
    h = (
        p.R
        + (p.M * (p.L + p.Lm) + 2.0 * p.mp * p.L) / denom * np.cos(theta)
        + (p.M * p.l) / denom * np.cos(phi)
    )
    return float(max(0.05, h))


def roll_ddot(
    theta: float,
    phi: float,
    x_dot: float,
    yaw_dot: float,
    roll: float,
    roll_dot: float,
    Froll: float,
    p: WheelLeggedParams,
) -> float:
    h = equivalent_height(theta, phi, p)
    mass_roll = p.M + 2.0 * p.mp
    gravity_torque = mass_roll * p.g * h * np.sin(roll)
    centrifugal_torque = mass_roll * x_dot * yaw_dot * h * np.cos(roll)
    support_torque = Froll * p.D / 2.0
    damping_torque = -p.Cpsi * roll_dot
    return float((gravity_torque + centrifugal_torque + support_torque + damping_torque) / p.Ix)


def height_alpha_ddot(
    alpha: float,
    alpha_dot: float,
    Fheight: float,
    leg: VirtualLegKinematics,
    p: WheelLeggedParams,
) -> float:
    alpha = leg.clip_alpha(alpha)
    eta = p.eta
    mh = p.M + p.mp * eta**2
    gh = (p.M + p.mp * eta) * p.g
    j1 = leg.J1(alpha)
    j2 = leg.J2(alpha)
    if abs(j1) < 1e-6:
        return 0.0
    return float((Fheight - gh - mh * j2 * alpha_dot**2) / (mh * j1))
