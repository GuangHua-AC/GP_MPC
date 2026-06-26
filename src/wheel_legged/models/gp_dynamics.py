from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


class GPDynamicsModel:
    def __init__(self, state_dim: int, action_dim: int, active_dims: list[int] | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.active_dims = active_dims if active_dims is not None else list(range(state_dim))
        self.models: dict[int, GaussianProcessRegressor] = {}
        self.x_mean = None
        self.x_std = None

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, max_points: int = 1500, seed: int = 0) -> None:
        x = np.concatenate([states, actions], axis=1).astype(float)
        y = (next_states - states).astype(float)
        if len(x) > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(x), size=max_points, replace=False)
            x = x[idx]
            y = y[idx]
        self.x_mean = x.mean(axis=0, keepdims=True)
        self.x_std = x.std(axis=0, keepdims=True) + 1e-6
        x_norm = (x - self.x_mean) / self.x_std
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(
                length_scale=np.ones(x_norm.shape[1]),
                length_scale_bounds=(1e-3, 1e3),
            )
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
        )
        self.models = {}
        for dim in self.active_dims:
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                alpha=1e-6,
                n_restarts_optimizer=0,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gp.fit(x_norm, y[:, dim])
            self.models[dim] = gp

    def predict(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([states, actions], axis=1).astype(float)
        x_norm = (x - self.x_mean) / self.x_std
        delta = np.zeros((len(states), self.state_dim), dtype=np.float32)
        pred_std = np.zeros((len(states), self.state_dim), dtype=np.float32)
        for dim, gp in self.models.items():
            mean, std_dim = gp.predict(x_norm, return_std=True)
            delta[:, dim] = mean.astype(np.float32)
            pred_std[:, dim] = std_dim.astype(np.float32)
        return states + delta, pred_std

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "GPDynamicsModel":
        return joblib.load(path)
