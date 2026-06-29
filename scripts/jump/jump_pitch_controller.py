from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PitchTorqueDebug:
    raw: float
    clipped: float
    saturated: bool


class JumpPitchController:
    """Virtual pitch torque PD controller for the 2D smoke jump.

    This controller intentionally does not use left/right wheel force
    difference. Differential wheels primarily affect yaw/roll, while this
    side-view smoke uses a simplified virtual pitch torque before actuator
    mapping is introduced later.
    """

    def compute(
        self,
        theta: float,
        theta_dot: float,
        theta_ref: float,
        theta_dot_ref: float,
        p,
    ) -> PitchTorqueDebug:
        raw = p.Kp_theta * (theta_ref - theta) + p.Kd_theta * (theta_dot_ref - theta_dot)
        clipped = float(np.clip(raw, -p.T_pitch_max, p.T_pitch_max))
        return PitchTorqueDebug(
            raw=float(raw),
            clipped=clipped,
            saturated=abs(clipped - raw) > 1e-9,
        )
