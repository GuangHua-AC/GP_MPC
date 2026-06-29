from __future__ import annotations

from enum import IntEnum


class JumpPhase(IntEnum):
    SQUAT = 0
    THRUST = 1
    FLIGHT_UP = 2
    FLIGHT_DOWN = 3
    LANDING = 4
    RECOVER = 5


PHASE_NAMES = {
    JumpPhase.SQUAT: "SQUAT",
    JumpPhase.THRUST: "THRUST",
    JumpPhase.FLIGHT_UP: "FLIGHT_UP",
    JumpPhase.FLIGHT_DOWN: "FLIGHT_DOWN",
    JumpPhase.LANDING: "LANDING",
    JumpPhase.RECOVER: "RECOVER",
}


class JumpPhaseObserver:
    """Small 1D jump state machine observer.

    The PDF revision asks for phase logic rather than using the old support
    leg-height model through the whole jump. This observer only decides phase
    transitions; the model/controller own the dynamics.
    """

    def squat_done(self, t_phase: float, L: float, L_dot: float, p) -> bool:
        return t_phase >= p.squat_time and abs(L - p.L_squat) < p.L_tol and abs(L_dot) < p.L_dot_tol

    def takeoff_detected(self, t_phase: float, z_b_dot: float, L: float, p) -> bool:
        return t_phase >= p.min_thrust_time and (
            z_b_dot >= p.v_takeoff_target or t_phase >= p.thrust_time
        )

    def apex_detected(self, z_com_dot: float) -> bool:
        return z_com_dot <= 0.0

    def landing_detected(self, z_w: float, z_w_dot: float, p) -> bool:
        return z_w <= p.wheel_radius and z_w_dot < 0.0

    def landing_done(self, t_phase: float, z_b_dot: float, L: float, L_dot: float, p) -> bool:
        return (
            t_phase >= p.min_landing_time
            and abs(z_b_dot) < p.recover_z_dot_tol
            and abs(L - p.L_stand) < p.recover_L_tol
            and abs(L_dot) < p.recover_L_dot_tol
        )

    def recover_done(self, t_phase: float, z_b_dot: float, L: float, L_dot: float, p) -> bool:
        return (
            t_phase >= p.recover_time
            and abs(z_b_dot) < p.recover_z_dot_tol
            and abs(L - p.L_stand) < p.recover_L_tol
            and abs(L_dot) < p.recover_L_dot_tol
        )
