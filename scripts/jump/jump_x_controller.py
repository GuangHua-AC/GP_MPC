from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class XForceDebug:
    raw: float
    clipped: float
    saturated: bool


class JumpXController:
    """Simple contact-only horizontal virtual force controller."""

    def compute(self, x: float, xdot: float, contact: bool, p) -> XForceDebug:
        if not contact:
            return XForceDebug(0.0, 0.0, False)

        raw = p.Kp_x * (p.x_ref - x) + p.Kd_x * (p.xdot_ref - xdot)
        clipped = float(np.clip(raw, -p.Fx_max, p.Fx_max))
        return XForceDebug(
            raw=float(raw),
            clipped=clipped,
            saturated=abs(clipped - raw) > 1e-9,
        )
