from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from wheel_legged.controllers.pd import WheelLeggedPDController
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv


class PredictiveModel(Protocol):
    def predict(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return next states and optional predictive std."""


@dataclass
class MPCConfig:
    horizon: int = 15
    candidates: int = 256
    seed: int = 0
    noise_scale: float = 0.25
    random_fraction: float = 0.15
    uncertainty_weight: float = 0.0


class RandomShootingMPC:
    """Random-shooting MPC for learned dynamics models."""

    def __init__(
        self,
        env: WheelLeggedEnv,
        model: PredictiveModel,
        config: MPCConfig | None = None,
        guide: WheelLeggedPDController | None = None,
    ):
        self.env = env
        self.model = model
        self.cfg = config or MPCConfig()
        self.guide = guide or WheelLeggedPDController(env)
        self.rng = np.random.default_rng(self.cfg.seed)
        p = env.p
        self.low = np.array([-p.T_limit, -p.Tp_limit, -p.Tyaw_limit, -p.Froll_limit, p.Fheight_min, -p.leg_diff_cmd_limit])
        self.high = np.array([p.T_limit, p.Tp_limit, p.Tyaw_limit, p.Froll_limit, p.Fheight_max, p.leg_diff_cmd_limit])

    def _sample_sequences(self, state: np.ndarray, ref: Reference) -> np.ndarray:
        cfg = self.cfg
        base = self.guide.act(state, ref)
        scale = (self.high - self.low) * cfg.noise_scale
        seq = base.reshape(1, 1, -1) + self.rng.normal(0.0, scale, size=(cfg.candidates, cfg.horizon, self.env.action_dim))
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
        seq = self._sample_sequences(np.asarray(state, dtype=float), ref)
        states = np.repeat(np.asarray(state, dtype=np.float32).reshape(1, -1), self.cfg.candidates, axis=0)
        costs = np.zeros(self.cfg.candidates, dtype=np.float64)
        uncertainty_cost = np.zeros_like(costs)

        for t in range(self.cfg.horizon):
            actions = seq[:, t, :]
            states, std = self.model.predict(states, actions)
            for i in range(self.cfg.candidates):
                costs[i] += self.env.cost(states[i], actions[i], ref)
            if std is not None and self.cfg.uncertainty_weight > 0.0:
                u = np.linalg.norm(std, axis=1)
                uncertainty_cost += u
                costs += self.cfg.uncertainty_weight * u

        best = int(np.argmin(costs))
        return seq[best, 0].copy(), {
            "best_idx": best,
            "best_cost": float(costs[best]),
            "mean_cost": float(np.mean(costs)),
            "uncertainty_cost": float(uncertainty_cost[best]),
        }
