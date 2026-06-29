from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jump_phase_observer import JumpPhase


@dataclass
class ForceDebug:
    raw: float
    clipped: float
    saturated: bool
    Kp: float
    Kd: float


class JumpVMCController:
    """Virtual leg force controller for the 1D jump smoke.

    Positive F_leg extends the leg / pushes the body upward in support phases.
    The old PDF chapter-3 support leg-height model is only valid while the
    wheel is in contact with the ground. Flight phases return zero support
    force and are handled by ballistic COM dynamics in jump_1d_model.py.
    """

    def gains_for_phase(self, phase: JumpPhase, t_phase: float, p) -> tuple[float, float]:
        if phase == JumpPhase.LANDING:
            Kp = min(p.Kp_land_max, p.Kp_land0 + p.Kp_land_rate * t_phase)
            Kd = min(p.Kd_land_max, p.Kd_land0 + p.Kd_land_rate * t_phase)
            return Kp, Kd
        if phase == JumpPhase.RECOVER:
            return p.Kp_recover, p.Kd_recover
        return p.Kp_L, p.Kd_L

    def compute_leg_force(self, state, ref, phase: JumpPhase, t_phase: float, p) -> ForceDebug:
        if phase in {JumpPhase.FLIGHT_UP, JumpPhase.FLIGHT_DOWN}:
            return ForceDebug(0.0, 0.0, False, 0.0, 0.0)

        Kp, Kd = self.gains_for_phase(phase, t_phase, p)
        if phase == JumpPhase.LANDING:
            raw = (
                Kp * (ref.L - state.L)
                + Kd * (ref.L_dot - state.L_dot)
                + p.G_eff
            )
        else:
            raw = (
                p.M_eff
                * (
                    ref.L_ddot
                    + Kd * (ref.L_dot - state.L_dot)
                    + Kp * (ref.L - state.L)
                )
                + p.G_eff
            )
        clipped = float(np.clip(raw, p.F_min, p.F_max))
        return ForceDebug(
            raw=float(raw),
            clipped=clipped,
            saturated=abs(clipped - raw) > 1e-9,
            Kp=float(Kp),
            Kd=float(Kd),
        )
