from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from wheel_legged.controllers.pd import WheelLeggedPDController
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv
from wheel_legged.pmpc.chance_constraints import compute_chance_penalty
from wheel_legged.pmpc.risk_cost import safe_norm_std, terminal_state_cost


class PredictiveModel(Protocol):
    def predict(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return next states and optional predictive std."""


@dataclass
class RiskAwareMPCConfig:
    horizon: int = 8
    candidates: int = 128
    seed: int = 0
    noise_scale: float = 0.25
    random_fraction: float = 0.15
    uncertainty_weight: float = 5.0
    chance_weight: float = 50.0
    guide_weight: float = 0.0
    k_sigma: float = 2.0
    terminal_weight: float = 0.0
    chance_enabled: tuple[str, ...] = ("theta", "phi", "x")


class RiskAwareShootingMPC:
    """Random-shooting GP-PMPC v0 for pure Python state14/action6.

    This class intentionally does not use the PMPC reference-style action from
    the bridge layer. Current GP models were trained on pure Python action6:
    [T, Tp, Tyaw, Froll, Fheight, leg_diff_cmd].
    """

    def __init__(
        self,
        env: WheelLeggedEnv,
        model: PredictiveModel,
        config: RiskAwareMPCConfig | None = None,
        guide: WheelLeggedPDController | None = None,
    ):
        self.env = env
        self.model = model
        self.cfg = config or RiskAwareMPCConfig()
        self.guide = guide or WheelLeggedPDController(env)
        self.rng = np.random.default_rng(self.cfg.seed)
        p = env.p
        self.low = np.array(
            [-p.T_limit, -p.Tp_limit, -p.Tyaw_limit, -p.Froll_limit, p.Fheight_min, -p.leg_diff_cmd_limit],
            dtype=float,
        )
        self.high = np.array(
            [p.T_limit, p.Tp_limit, p.Tyaw_limit, p.Froll_limit, p.Fheight_max, p.leg_diff_cmd_limit],
            dtype=float,
        )

    def _sample_sequences(self, state: np.ndarray, ref: Reference) -> np.ndarray:
        cfg = self.cfg
        base = self.guide.act(state, ref)
        scale = (self.high - self.low) * cfg.noise_scale
        seq = base.reshape(1, 1, -1) + self.rng.normal(
            0.0,
            scale,
            size=(cfg.candidates, cfg.horizon, self.env.action_dim),
        )
        seq = np.clip(seq, self.low, self.high)
        seq[0, :, :] = base

        n_random = int(cfg.candidates * cfg.random_fraction)
        if n_random > 0:
            seq[-n_random:] = self.rng.uniform(self.low, self.high, size=(n_random, cfg.horizon, self.env.action_dim))

        for k in range(seq.shape[0]):
            for t in range(1, seq.shape[1]):
                seq[k, t] = 0.65 * seq[k, t - 1] + 0.35 * seq[k, t]
            for t in range(seq.shape[1]):
                seq[k, t] = self.env._clip_action(seq[k, t])
        return seq.astype(np.float32)

    def plan(self, state: np.ndarray, ref: Reference | None = None) -> tuple[np.ndarray, dict]:
        ref = ref or self.env.ref
        cfg = self.cfg
        state_arr = np.asarray(state, dtype=float)
        guide_action = self.guide.act(state_arr, ref)
        seq = self._sample_sequences(np.asarray(state, dtype=float), ref)
        states = np.repeat(np.asarray(state, dtype=np.float32).reshape(1, -1), cfg.candidates, axis=0)
        costs = np.zeros(cfg.candidates, dtype=np.float64)
        tracking_cost = np.zeros(cfg.candidates, dtype=np.float64)
        uncertainty_cost = np.zeros(cfg.candidates, dtype=np.float64)
        chance_penalty = np.zeros(cfg.candidates, dtype=np.float64)
        terminal_cost = np.zeros(cfg.candidates, dtype=np.float64)
        guide_cost = np.zeros(cfg.candidates, dtype=np.float64)
        use_uncertainty = cfg.uncertainty_weight > 0.0
        use_chance = cfg.chance_weight > 0.0
        use_terminal = cfg.terminal_weight > 0.0
        use_guide = cfg.guide_weight > 0.0
        action_range = np.maximum(self.high - self.low, 1e-6)

        for t in range(cfg.horizon):
            actions = seq[:, t, :]
            states, std = self.model.predict(states, actions)
            step_cost = np.asarray([self.env.cost(states[i], actions[i], ref) for i in range(cfg.candidates)])
            tracking_cost += step_cost
            costs += step_cost

            if use_uncertainty or use_chance:
                std_arr = np.zeros_like(states) if std is None else np.nan_to_num(std, nan=0.0, posinf=1e6, neginf=0.0)
                if use_uncertainty:
                    u_cost = safe_norm_std(std_arr)
                    uncertainty_cost += u_cost
                    costs += cfg.uncertainty_weight * u_cost
                if use_chance:
                    c_penalty = compute_chance_penalty(
                        states,
                        std_arr,
                        self.env,
                        ref,
                        cfg.k_sigma,
                        enabled=cfg.chance_enabled,
                    )
                    chance_penalty += c_penalty
                    costs += cfg.chance_weight * c_penalty
            if use_guide:
                g_cost = np.sum(((actions - guide_action) / action_range) ** 2, axis=1)
                guide_cost += g_cost
                costs += cfg.guide_weight * g_cost

        if use_terminal:
            terminal_cost = np.asarray([terminal_state_cost(states[i], self.env, ref) for i in range(cfg.candidates)])
            costs += cfg.terminal_weight * terminal_cost

        best = int(np.argmin(costs))
        return seq[best, 0].copy(), {
            "best_idx": best,
            "best_cost": float(costs[best]),
            "mean_cost": float(np.mean(costs)),
            "best_first_action": seq[best, 0].copy(),
            "guide_action": guide_action.copy(),
            "best_cost_tracking": float(tracking_cost[best]),
            "best_cost_uncertainty": float(uncertainty_cost[best]),
            "best_cost_chance": float(chance_penalty[best]),
            "best_cost_terminal": float(terminal_cost[best]),
            "best_cost_guide": float(guide_cost[best]),
            "uncertainty_cost": float(uncertainty_cost[best]),
            "chance_penalty": float(chance_penalty[best]),
            "terminal_cost": float(terminal_cost[best]),
            "guide_cost": float(guide_cost[best]),
            "uncertainty_weight": float(cfg.uncertainty_weight),
            "chance_weight": float(cfg.chance_weight),
            "guide_weight": float(cfg.guide_weight),
            "terminal_weight": float(cfg.terminal_weight),
            "max_abs_action": float(np.max(np.abs(seq[best, 0]))),
        }
