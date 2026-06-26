from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning


class NNDynamicsModel:
    """Neural-network dynamics model using sklearn's MLPRegressor.

    The model predicts state delta: next_state - state.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.device = device
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden, hidden),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            batch_size="auto",
            max_iter=200,
            warm_start=False,
            random_state=0,
        )
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, epochs: int = 80, lr: float = 1e-3) -> list[float]:
        self.model.learning_rate_init = lr
        self.model.max_iter = epochs
        self.model.warm_start = False
        x = np.concatenate([states, actions], axis=1).astype(np.float32)
        y = (next_states - states).astype(np.float32)
        self.x_mean = x.mean(axis=0, keepdims=True)
        self.x_std = x.std(axis=0, keepdims=True) + 1e-6
        self.y_mean = y.mean(axis=0, keepdims=True)
        self.y_std = y.std(axis=0, keepdims=True) + 1e-6
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(x_norm, y_norm)
        if hasattr(self.model, "loss_curve_"):
            return [float(v) for v in self.model.loss_curve_]
        if hasattr(self.model, "loss_"):
            return [float(self.model.loss_)]
        return [float("nan")]

    def predict(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, None]:
        x = np.concatenate([states, actions], axis=1).astype(np.float32)
        x_norm = (x - self.x_mean) / self.x_std
        delta = self.model.predict(x_norm) * self.y_std + self.y_mean
        return states + delta.astype(np.float32), None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "NNDynamicsModel":
        obj = joblib.load(path)
        obj.device = device
        return obj
