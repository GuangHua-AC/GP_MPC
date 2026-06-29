from __future__ import annotations

import numpy as np


def _finite_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)


def python_state14_to_reduced(state, env=None) -> np.ndarray:
    """Convert current pure Python state14 to PMPC reduced_state14.

    Input state:
        [theta, theta_dot, x, x_dot, phi, phi_dot, yaw, yaw_dot,
         roll, roll_dot, alpha, alpha_dot, leg_diff, leg_diff_dot]

    Output state:
        [theta, theta_dot, x, x_dot, phi, phi_dot, yaw, yaw_dot,
         roll, roll_dot, L0_mean, L0_mean_dot, leg_diff, leg_diff_dot]

    If env is None, alpha and alpha_dot are copied into L0_mean and
    L0_mean_dot as a smoke-test fallback only. Real PMPC/Isaac data should use
    env.leg.L0(alpha) and env.leg.L0_dot(alpha, alpha_dot).
    """

    s = _finite_array(np.asarray(state, dtype=float).reshape(-1))
    if s.size != 14:
        raise ValueError(f"expected state shape (14,), got {s.shape}")

    reduced = s.copy()
    alpha = float(s[10])
    alpha_dot = float(s[11])
    if env is not None:
        L0_mean = float(env.leg.L0(alpha))
        L0_mean_dot = float(env.leg.L0_dot(alpha, alpha_dot))
    else:
        # Smoke fallback only: without env kinematics alpha is kept as the
        # placeholder for L0_mean. Do not use this for Isaac or final PMPC data.
        L0_mean = alpha
        L0_mean_dot = alpha_dot

    reduced[10] = L0_mean
    reduced[11] = L0_mean_dot
    reduced = _finite_array(reduced)
    if reduced.shape != (14,):
        raise RuntimeError(f"reduced state shape error: {reduced.shape}")
    return reduced
