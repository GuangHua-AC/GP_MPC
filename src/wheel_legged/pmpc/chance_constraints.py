from __future__ import annotations

import numpy as np

from wheel_legged.dynamics.terrain import terrain_heights


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)


def _constraint_penalty(
    mu: np.ndarray,
    sigma: np.ndarray,
    center: float,
    limit: float,
    k_sigma: float,
    warning_ratio: float,
) -> np.ndarray:
    risk = (np.abs(mu - center) + k_sigma * np.maximum(sigma, 0.0)) / max(limit, 1e-9)
    early_warning = np.maximum(0.0, risk - warning_ratio) ** 2
    hard_violation = np.maximum(0.0, risk - 1.0) ** 2
    return early_warning + 10.0 * hard_violation


def compute_chance_penalty(
    states_mu,
    states_std,
    env,
    ref,
    k_sigma: float = 2.0,
    enabled: tuple[str, ...] = ("theta", "phi", "x"),
    warning_ratio: float = 0.25,
) -> np.ndarray:
    """Diagonal chance-constraint penalty for current PMPC v0.

    Constraint form:
        risk = (abs(mu - center) + k*sigma) / limit
        penalty = max(0, risk - warning_ratio)^2 + 10 * max(0, risk - 1)^2

    The early-warning term is intentionally a PMPC planning margin, not an env
    termination limit. It makes chance costs visible before the short horizon is
    already at the fall boundary.

    Missing or zero std dimensions are allowed. This is important for current
    balance GP models, which may only train active state dimensions.
    """

    mu = _as_2d(states_mu)
    std = _as_2d(states_std)
    if std.shape != mu.shape:
        std_full = np.zeros_like(mu)
        cols = min(std.shape[1], mu.shape[1])
        std_full[:, :cols] = std[:, :cols]
        std = std_full

    p = env.p
    x_limit = ref.x_limit if ref.x_limit is not None else p.max_abs_x
    roll_limit = min(p.max_abs_roll, 0.06) if getattr(env, "task", "") == "terrain" else p.max_abs_roll
    alpha = np.asarray([env.leg.clip_alpha(a) for a in mu[:, 10]], dtype=float)
    l0_mu = np.asarray([env.leg.L0(a) for a in alpha], dtype=float)
    l0_std = np.abs(np.asarray([env.leg.J1(a) for a in alpha], dtype=float)) * np.maximum(std[:, 10], 0.0)
    leg_diff_center = np.zeros(mu.shape[0], dtype=float)
    if getattr(env, "task", "") == "terrain" and getattr(env, "terrain_known_to_controller", True):
        terrain_diffs = []
        for x_value in mu[:, 2]:
            left_h, right_h = terrain_heights(float(x_value), env.terrain_mode, p)
            terrain_diffs.append(left_h - right_h)
        terrain_diffs = np.asarray(terrain_diffs, dtype=float)
        leg_diff_center = -terrain_diffs

    term_map = {
        "theta": _constraint_penalty(mu[:, 0], std[:, 0], 0.0, p.max_abs_theta, k_sigma, warning_ratio),
        "phi": _constraint_penalty(mu[:, 4], std[:, 4], 0.0, p.max_abs_phi, k_sigma, warning_ratio),
        "x": _constraint_penalty(mu[:, 2], std[:, 2], ref.x_ref, x_limit, k_sigma, warning_ratio),
        "L0": _constraint_penalty(l0_mu, l0_std, ref.L0_ref, 0.08, k_sigma, warning_ratio),
        "alpha": _constraint_penalty(mu[:, 10], std[:, 10], env.leg.alpha_from_L0(ref.L0_ref), 0.8, k_sigma, warning_ratio),
        "yaw": _constraint_penalty(mu[:, 6], std[:, 6], 0.0, p.max_abs_yaw, k_sigma, warning_ratio),
        "roll": _constraint_penalty(mu[:, 8], std[:, 8], ref.roll_ref, roll_limit, k_sigma, warning_ratio),
        "leg_diff": _constraint_penalty(mu[:, 12], std[:, 12], leg_diff_center, p.leg_diff_limit, k_sigma, warning_ratio),
    }
    terms = [term_map[name] for name in enabled if name in term_map]
    if not terms:
        return np.zeros(mu.shape[0], dtype=float)
    penalty = np.sum(np.vstack(terms), axis=0)
    return np.nan_to_num(penalty, nan=0.0, posinf=1e12, neginf=0.0)
