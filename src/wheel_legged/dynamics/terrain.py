from __future__ import annotations

import numpy as np

from .parameters import WheelLeggedParams


def smooth_box(x: float, start: float, end: float, edge: float) -> float:
    up = 0.5 * (1.0 + np.tanh((x - start) / max(edge, 1e-6)))
    down = 0.5 * (1.0 - np.tanh((x - end) / max(edge, 1e-6)))
    return float(up * down)


def terrain_heights(x: float, mode: str, p: WheelLeggedParams) -> tuple[float, float]:
    if mode == "flat":
        return 0.0, 0.0

    if mode in {"left_obstacle", "right_obstacle"}:
        shape = smooth_box(x, p.obstacle_start, p.obstacle_start + p.obstacle_length, p.obstacle_edge)
        height = float(p.obstacle_height * shape)
        if mode == "left_obstacle":
            return height, 0.0
        return 0.0, height

    if mode == "sine":
        diff = float(p.obstacle_height * np.sin(2.0 * np.pi * x / p.terrain_wavelength))
        return 0.5 * diff, -0.5 * diff

    if mode == "panorama":
        left_step = p.obstacle_height * smooth_box(x, 4.55, 4.95, p.obstacle_edge)
        right_step = 0.75 * p.obstacle_height * smooth_box(x, 5.22, 5.60, p.obstacle_edge)
        return float(left_step), float(right_step)

    raise ValueError(f"unknown terrain mode: {mode}")
