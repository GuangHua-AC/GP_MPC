from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from wheel_legged.controllers.pd import WheelLeggedPDController


@dataclass(frozen=True)
class PMPCActionBounds:
    theta0_mean_ref: tuple[float, float] = (-0.5, 0.5)
    theta0_diff_ref: tuple[float, float] = (-0.4, 0.4)
    L0_mean_ref: tuple[float, float] = (0.24, 0.40)
    L0_diff_ref: tuple[float, float] = (-0.08, 0.08)
    wheel_vel_mean_ref: tuple[float, float] = (-8.0, 8.0)
    wheel_vel_diff_ref: tuple[float, float] = (-4.0, 4.0)

    isaac_theta0_ref: tuple[float, float] = (-0.7, 0.7)
    isaac_l0_ref: tuple[float, float] = (0.20, 0.44)
    isaac_wheel_vel_ref: tuple[float, float] = (-10.0, 10.0)

    def lower(self) -> np.ndarray:
        return np.array(
            [
                self.theta0_mean_ref[0],
                self.theta0_diff_ref[0],
                self.L0_mean_ref[0],
                self.L0_diff_ref[0],
                self.wheel_vel_mean_ref[0],
                self.wheel_vel_diff_ref[0],
            ],
            dtype=float,
        )

    def upper(self) -> np.ndarray:
        return np.array(
            [
                self.theta0_mean_ref[1],
                self.theta0_diff_ref[1],
                self.L0_mean_ref[1],
                self.L0_diff_ref[1],
                self.wheel_vel_mean_ref[1],
                self.wheel_vel_diff_ref[1],
            ],
            dtype=float,
        )

    def contains(self, u_pmpc: np.ndarray) -> bool:
        u = np.asarray(u_pmpc, dtype=float).reshape(6)
        return bool(np.all(u >= self.lower()) and np.all(u <= self.upper()))


def _bounds(bounds: PMPCActionBounds | None) -> PMPCActionBounds:
    return bounds or PMPCActionBounds()


def _finite_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)


def clip_pmpc_action6(u_pmpc, bounds: PMPCActionBounds | None = None) -> np.ndarray:
    b = _bounds(bounds)
    u = _finite_array(np.asarray(u_pmpc, dtype=float).reshape(-1))
    if u.size != 6:
        raise ValueError(f"expected PMPC action shape (6,), got {u.shape}")
    clipped = np.clip(u, b.lower(), b.upper())
    clipped = _finite_array(clipped)
    if clipped.shape != (6,):
        raise RuntimeError(f"clipped PMPC action shape error: {clipped.shape}")
    return clipped


def pmpc_action6_to_isaac_vmc_action6(u_pmpc, bounds: PMPCActionBounds | None = None) -> np.ndarray:
    b = _bounds(bounds)
    u = clip_pmpc_action6(u_pmpc, b)
    (
        theta0_mean_ref,
        theta0_diff_ref,
        L0_mean_ref,
        L0_diff_ref,
        wheel_vel_mean_ref,
        wheel_vel_diff_ref,
    ) = u

    left_theta0_ref = theta0_mean_ref + 0.5 * theta0_diff_ref
    right_theta0_ref = theta0_mean_ref - 0.5 * theta0_diff_ref
    left_l0_ref = L0_mean_ref + 0.5 * L0_diff_ref
    right_l0_ref = L0_mean_ref - 0.5 * L0_diff_ref
    left_wheel_vel = wheel_vel_mean_ref + 0.5 * wheel_vel_diff_ref
    right_wheel_vel = wheel_vel_mean_ref - 0.5 * wheel_vel_diff_ref

    action = np.array(
        [
            left_theta0_ref,
            left_l0_ref,
            left_wheel_vel,
            right_theta0_ref,
            right_l0_ref,
            right_wheel_vel,
        ],
        dtype=float,
    )
    lower = np.array(
        [
            b.isaac_theta0_ref[0],
            b.isaac_l0_ref[0],
            b.isaac_wheel_vel_ref[0],
            b.isaac_theta0_ref[0],
            b.isaac_l0_ref[0],
            b.isaac_wheel_vel_ref[0],
        ],
        dtype=float,
    )
    upper = np.array(
        [
            b.isaac_theta0_ref[1],
            b.isaac_l0_ref[1],
            b.isaac_wheel_vel_ref[1],
            b.isaac_theta0_ref[1],
            b.isaac_l0_ref[1],
            b.isaac_wheel_vel_ref[1],
        ],
        dtype=float,
    )
    return _finite_array(np.clip(action, lower, upper))


def pmpc_action6_to_python_action6(u_pmpc, state, ref, env, bounds: PMPCActionBounds | None = None):
    """Map PMPC action to pure Python action6 using PD as nominal guide.

    This first bridge version intentionally avoids claiming a full physical
    mapping from PMPC references to force-style Python actions. It uses the
    existing WheelLeggedPDController as the nominal guide and only routes the
    immediately meaningful dimensions into the Reference object.
    """

    b = _bounds(bounds)
    u = clip_pmpc_action6(u_pmpc, b)
    (
        theta0_mean_ref,
        theta0_diff_ref,
        L0_mean_ref,
        L0_diff_ref,
        wheel_vel_mean_ref,
        wheel_vel_diff_ref,
    ) = u

    wheel_radius = float(getattr(env.p, "R", 0.08))
    wheel_base = max(float(getattr(env.p, "D", 0.40)), 1e-6)
    v_ref = wheel_radius * wheel_vel_mean_ref
    yaw_delta = 0.20 * wheel_radius * wheel_vel_diff_ref / wheel_base

    adapted_ref = replace(
        ref,
        v_ref=float(v_ref),
        yaw_ref=float(ref.yaw_ref + yaw_delta),
        L0_ref=float(L0_mean_ref),
    )
    controller = WheelLeggedPDController(env)
    action = controller.act(np.asarray(state, dtype=float), adapted_ref)
    action = _finite_array(np.asarray(action, dtype=float).reshape(6))

    info = {
        "clipped_pmpc_action6": u.copy(),
        "adapted_ref": {
            "v_ref": float(adapted_ref.v_ref),
            "yaw_ref": float(adapted_ref.yaw_ref),
            "L0_ref": float(adapted_ref.L0_ref),
            "L0_diff_ref": float(L0_diff_ref),
        },
        "used_dimensions": [
            "L0_mean_ref -> ref.L0_ref",
            "wheel_vel_mean_ref -> ref.v_ref using env.p.R",
            "wheel_vel_diff_ref -> small yaw_ref offset using env.p.R/env.p.D",
        ],
        "reserved_dimensions": [
            "theta0_mean_ref",
            "theta0_diff_ref",
            "L0_diff_ref for direct pure Python force mapping",
        ],
        "reserved_values": {
            "theta0_mean_ref": float(theta0_mean_ref),
            "theta0_diff_ref": float(theta0_diff_ref),
            "L0_diff_ref": float(L0_diff_ref),
        },
        "mapping_status": "smoke_pd_nominal_guide",
    }
    return action, info
