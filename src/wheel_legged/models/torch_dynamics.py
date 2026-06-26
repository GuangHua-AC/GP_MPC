from __future__ import annotations

from pathlib import Path

import numpy as np


class TorchDynamicsModel:
    """PyTorch MLP dynamics model.

    The model predicts state delta: next_state - state. Importing torch is
    intentionally delayed until construction so the rest of the project still
    works when PyTorch is not installed.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        device: str = "auto",
        batch_size: int = 2048,
    ):
        import torch
        from torch import nn

        self.torch = torch
        self.nn = nn
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.batch_size = batch_size

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        ).to(self.device)

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        epochs: int = 120,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> list[float]:
        torch = self.torch
        x = np.concatenate([states, actions], axis=1).astype(np.float32)
        y = (next_states - states).astype(np.float32)

        self.x_mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.x_std = (x.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        self.y_mean = y.mean(axis=0, keepdims=True).astype(np.float32)
        self.y_std = (y.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        x_t = torch.from_numpy(x_norm)
        y_t = torch.from_numpy(y_norm)

        dataset = torch.utils.data.TensorDataset(x_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = self.nn.MSELoss()
        losses: list[float] = []

        for epoch in range(epochs):
            self.model.train()
            total = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * len(xb)
                count += len(xb)

            epoch_loss = total / max(count, 1)
            losses.append(epoch_loss)
            if verbose:
                print(f"epoch={epoch + 1:04d}/{epochs} loss={epoch_loss:.6f} device={self.device}")

        self.model.eval()
        return losses

    def predict(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, None]:
        torch = self.torch
        x = np.concatenate([states, actions], axis=1).astype(np.float32)
        x_norm = (x - self.x_mean) / self.x_std
        with torch.no_grad():
            xt = torch.from_numpy(x_norm).to(self.device)
            delta_norm = self.model(xt).cpu().numpy()
        delta = delta_norm * self.y_std + self.y_mean
        return states + delta.astype(np.float32), None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "backend": "torch",
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden": self.hidden,
                "batch_size": self.batch_size,
                "state_dict": self.model.state_dict(),
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "TorchDynamicsModel":
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            ckpt["state_dim"],
            ckpt["action_dim"],
            hidden=ckpt["hidden"],
            device=device,
            batch_size=ckpt.get("batch_size", 2048),
        )
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.x_mean = ckpt["x_mean"]
        obj.x_std = ckpt["x_std"]
        obj.y_mean = ckpt["y_mean"]
        obj.y_std = ckpt["y_std"]
        obj.model.eval()
        return obj

