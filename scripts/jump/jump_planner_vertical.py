from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jump_phase_observer import JumpPhase


@dataclass
class LegRef:
    L: float
    L_dot: float
    L_ddot: float


def cubic_segment(t: float, duration: float, y0: float, y1: float) -> LegRef:
    """Cubic smoothstep segment with zero endpoint velocity."""
    T = max(float(duration), 1e-6)
    tau = float(np.clip(t / T, 0.0, 1.0))
    dy = y1 - y0
    s = 3.0 * tau**2 - 2.0 * tau**3
    s_dot = (6.0 * tau - 6.0 * tau**2) / T
    s_ddot = (6.0 - 12.0 * tau) / (T**2)
    return LegRef(y0 + dy * s, dy * s_dot, dy * s_ddot)


class VerticalJumpPlanner:
    """Leg-length reference generator for the 1D smoke jump."""

    def get_ref(self, phase: JumpPhase, t_phase: float, state, p) -> LegRef:
        if phase == JumpPhase.SQUAT:
            return cubic_segment(t_phase, p.squat_time, p.L_stand, p.L_squat)

        if phase == JumpPhase.THRUST:
            return cubic_segment(t_phase, p.thrust_time, p.L_squat, p.L_takeoff)

        if phase == JumpPhase.FLIGHT_UP:
            return cubic_segment(t_phase, p.flight_retract_time, p.L_takeoff, p.L_air_min)

        if phase == JumpPhase.FLIGHT_DOWN:
            return cubic_segment(t_phase, p.flight_extend_time, p.L_air_min, p.L_landing)

        if phase == JumpPhase.LANDING:
            # Landing uses high damping in the controller; keep the target simple
            # and let the impedance absorb vertical speed.
            return LegRef(p.L_stand, 0.0, 0.0)

        if phase == JumpPhase.RECOVER:
            return LegRef(p.L_stand, 0.0, 0.0)

        raise ValueError(f"unknown jump phase: {phase}")

    def flight_ref_from_lengths(
        self,
        phase: JumpPhase,
        t_phase: float,
        start_L: float,
        target_L: float,
        p,
    ) -> LegRef:
        if phase == JumpPhase.FLIGHT_UP:
            return cubic_segment(t_phase, p.flight_retract_time, start_L, target_L)
        if phase == JumpPhase.FLIGHT_DOWN:
            return cubic_segment(t_phase, p.flight_extend_time, start_L, target_L)
        return self.get_ref(phase, t_phase, None, p)
