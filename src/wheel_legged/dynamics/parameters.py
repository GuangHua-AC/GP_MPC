from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WheelLeggedParams:
    dt: float = 0.005
    max_steps: int = 1200
    g: float = 9.81

    R: float = 0.08
    D: float = 0.40
    L: float = 0.28
    Lm: float = 0.08
    l: float = 0.12
    h: float = 0.55

    mw: float = 0.40
    mp: float = 0.30
    M: float = 3.00

    Iw: float = 0.002
    Ip: float = 0.030
    Im: float = 0.050
    Iz: float = 0.060
    Ix: float = 0.045
    Cpsi: float = 0.18

    eta: float = 0.55
    L0_offset: float = 0.30
    L0_gain: float = 0.10
    alpha_min: float = -0.90
    alpha_max: float = 0.90

    T_limit: float = 1.20
    Tp_limit: float = 1.20
    Tyaw_limit: float = 0.80
    Froll_limit: float = 90.0
    Fheight_min: float = 0.0
    Fheight_max: float = 140.0

    leg_diff_limit: float = 0.12
    leg_diff_cmd_limit: float = 0.12
    leg_diff_omega: float = 24.0
    leg_diff_zeta: float = 0.85

    obstacle_height: float = 0.04
    obstacle_start: float = 0.35
    obstacle_length: float = 0.35
    obstacle_edge: float = 0.035
    terrain_wavelength: float = 1.2

    max_abs_theta: float = 0.80
    max_abs_phi: float = 0.80
    max_abs_roll: float = 0.60
    max_abs_x: float = 5.0
    max_abs_yaw: float = 2.50

