from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class JumpParams:
    """Central parameter set for the pure Python 1D vertical jump smoke."""

    g: float = 9.81
    m_body: float = 8.0
    m_wheel: float = 2.0
    wheel_radius: float = 0.10

    L_min: float = 0.20
    L_squat: float = 0.24
    L_stand: float = 0.36
    L_takeoff: float = 0.40
    L_air_min: float = 0.22
    L_landing: float = 0.34

    F_min: float = 0.0
    F_max: float = 450.0
    Kp_L: float = 1200.0
    Kd_L: float = 80.0
    contact_eps: float = 5.0
    h_target: float = 0.03

    dt: float = 0.001
    t_final: float = 3.0
    squat_time: float = 0.35
    thrust_time: float = 0.22
    min_thrust_time: float = 0.0
    flight_retract_time: float = 0.16
    flight_extend_time: float = 0.20
    min_landing_time: float = 0.12
    recover_time: float = 0.25

    Kp_land0: float = 180.0
    Kp_land_rate: float = 700.0
    Kp_land_max: float = 700.0
    Kd_land0: float = 260.0
    Kd_land_rate: float = 80.0
    Kd_land_max: float = 340.0
    Kp_recover: float = 700.0
    Kd_recover: float = 130.0

    L_tol: float = 0.008
    L_dot_tol: float = 0.08
    recover_z_dot_tol: float = 0.08
    recover_L_tol: float = 0.012
    recover_L_dot_tol: float = 0.10
    takeoff_L_tol: float = 0.012

    Iyy: float = 0.35
    damping_theta: float = 0.15
    Kp_theta: float = 35.0
    Kd_theta: float = 6.0
    T_pitch_max: float = 8.0
    theta_ref: float = 0.0
    theta_dot_ref: float = 0.0
    disturbance_torque: float = 0.0

    Kp_x: float = 45.0
    Kd_x: float = 16.0
    Fx_max: float = 12.0
    x_ref: float = 0.0
    xdot_ref: float = 0.0
    x_force_offset: float = 0.01

    @property
    def M_total(self) -> float:
        return self.m_body + self.m_wheel

    @property
    def M_eff(self) -> float:
        return self.m_body

    @property
    def G_eff(self) -> float:
        return self.m_body * self.g

    @property
    def v_takeoff_target(self) -> float:
        return float(np.sqrt(2.0 * self.g * max(self.h_target, 0.0)))

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        return {f"param_{key}": np.array([value]) for key, value in asdict(self).items()}
