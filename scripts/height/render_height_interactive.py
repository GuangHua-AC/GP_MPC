from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np
import plotly.graph_objects as go


@dataclass
class Params:
    R: float = 0.08
    D: float = 0.40
    l1: float = 0.18
    l3: float = 0.22
    l5: float = 0.14
    body_dims: tuple[float, float, float] = (0.24, 0.20, 0.10)
    motor_l: float = 0.04
    dt: float = 0.005


def scalar(data, key: str, default=None):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def rotation_matrices(yaw: float, roll: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    return rz @ rx @ ry


def solve_two_link_ik(target_x, target_z, upper, lower, knee_dir):
    dist2 = target_x**2 + target_z**2
    dist = np.sqrt(max(dist2, 1e-12))
    max_len = upper + lower - 1e-4
    min_len = abs(upper - lower) + 1e-4
    if dist > max_len:
        scale = max_len / dist
        target_x *= scale
        target_z *= scale
        dist = max_len
        dist2 = dist**2
    elif dist < min_len:
        scale = min_len / dist
        target_x *= scale
        target_z *= scale
        dist = min_len
        dist2 = dist**2
    cos_alpha = np.clip((upper**2 + dist2 - lower**2) / (2.0 * upper * dist), -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    base = np.arctan2(target_z, target_x)
    q = base + knee_dir * alpha
    return upper * np.cos(q), upper * np.sin(q)


def line_trace(points, name: str, color: str, width: float = 5.0):
    pts = np.asarray(points, dtype=float)
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        showlegend=False,
    )


def marker_trace(points, name: str, color: str, size: float = 4.0):
    pts = np.asarray(points, dtype=float)
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=dict(color=color, size=size),
        name=name,
        showlegend=False,
    )


def wheel_circle(center: np.ndarray, radius: float, side: int, n: int = 40):
    ang = np.linspace(0.0, 2.0 * np.pi, n)
    pts = np.column_stack(
        [
            center[0] + radius * np.cos(ang),
            np.full_like(ang, center[1]),
            center[2] + radius * np.sin(ang),
        ]
    )
    return pts


def body_box(center: np.ndarray, rot: np.ndarray, size: tuple[float, float, float]):
    length, width, height = size
    dx, dy, dz = length / 2.0, width / 2.0, height / 2.0
    corners = np.array(
        [
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
        ],
        dtype=float,
    )
    verts = np.array([center + rot @ c for c in corners])
    order = [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]
    return verts[order]


def frame_traces(states: np.ndarray, l0s: np.ndarray, l0_refs: np.ndarray, idx: int, params: Params):
    theta, _theta_dot, x, _x_dot, phi, _phi_dot = states[idx, :6]
    l0 = l0s[idx]
    l0_ref = l0_refs[idx]
    rot_body = rotation_matrices(0.0, 0.0, phi)
    wheel_axis = np.array([x, 0.0, params.R], dtype=float)
    body_center = wheel_axis + np.array([params.R * np.sin(theta), 0.0, l0], dtype=float)
    traces = [
        line_trace(body_box(body_center, rot_body, params.body_dims), "body", "#777777", 4.0),
    ]
    wheel_centers = []
    link_points = []
    motor_points = []
    dy_body = params.body_dims[1] / 2.0
    dx_motor = params.l5 / 2.0
    for side in [1, -1]:
        wheel_center = wheel_axis + np.array([0.0, side * params.D / 2.0, 0.0])
        wheel_centers.append(wheel_center)
        traces.append(line_trace(wheel_circle(wheel_center, params.R, side), "wheel", "#111111", 6.0))

        leg_y = side * (dy_body + params.motor_l)
        vec_local = rot_body.T @ (wheel_center - body_center)
        foot_x, foot_z = vec_local[0], vec_local[2]
        knee_a_x, knee_a_z = solve_two_link_ik(foot_x - dx_motor, foot_z, params.l1, params.l3, knee_dir=1.0)
        knee_e_x, knee_e_z = solve_two_link_ik(foot_x + dx_motor, foot_z, params.l1, params.l3, knee_dir=-1.0)

        a_tip = body_center + rot_body @ np.array([dx_motor, leg_y, 0.0])
        knee_a = body_center + rot_body @ np.array([dx_motor + knee_a_x, leg_y, knee_a_z])
        e_tip = body_center + rot_body @ np.array([-dx_motor, leg_y, 0.0])
        knee_e = body_center + rot_body @ np.array([-dx_motor + knee_e_x, leg_y, knee_e_z])
        foot_tip = wheel_center
        motor_points.extend([a_tip, e_tip])
        link_points.extend([a_tip, knee_a, foot_tip, [np.nan, np.nan, np.nan], e_tip, knee_e, foot_tip, [np.nan, np.nan, np.nan]])

    traces.append(line_trace(link_points, "five-bar links", "#000000", 7.0))
    traces.append(marker_trace(motor_points + wheel_centers, "joints", "#1f77b4", 5.0))
    traces.append(
        line_trace(
            [
                body_center + np.array([0.0, 0.0, 0.11]),
                body_center + np.array([0.0, 0.0, 0.11 + (l0_ref - l0)]),
            ],
            "height error",
            "#2ca02c",
            5.0,
        )
    )
    path = states[: idx + 1, 2]
    traces.append(line_trace(np.column_stack([path, np.zeros_like(path), np.zeros_like(path)]), "path", "#1f77ff", 4.0))
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=20)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    states = np.asarray(data["states"], dtype=float)
    l0s = np.asarray(data["L0s"], dtype=float).reshape(-1) if "L0s" in data.files else 0.30 + 0.10 * np.sin(states[:, 10])
    l0_refs = np.asarray(data["L0_refs"], dtype=float).reshape(-1) if "L0_refs" in data.files else np.zeros(len(states))
    dt = float(scalar(data, "dt", 0.005))
    params = Params(dt=dt)
    frames_idx = list(range(0, len(states), max(1, args.stride)))
    if frames_idx[-1] != len(states) - 1:
        frames_idx.append(len(states) - 1)

    first = frame_traces(states, l0s, l0_refs, frames_idx[0], params)
    frames = [
        go.Frame(
            data=frame_traces(states, l0s, l0_refs, idx, params),
            name=str(idx),
            layout=go.Layout(title_text=f"Height interactive view | step={idx} | t={idx * dt:.2f}s"),
        )
        for idx in frames_idx
    ]
    max_x = float(np.max(states[:, 2])) if len(states) else 1.0
    min_x = float(np.min(states[:, 2])) if len(states) else 0.0
    fig = go.Figure(data=first, frames=frames)
    fig.update_layout(
        title=f"Height interactive view | step={frames_idx[0]} | drag to rotate",
        scene=dict(
            xaxis=dict(title="World X / m", range=[min_x - 0.25, max_x + 0.25]),
            yaxis=dict(title="World Y / m", range=[-0.45, 0.45]),
            zaxis=dict(title="Z / m", range=[0.0, 0.65]),
            aspectmode="manual",
            aspectratio=dict(x=1.5, y=1.0, z=0.9),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate", args=[[str(idx)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=str(idx))
                    for idx in frames_idx
                ],
                currentvalue=dict(prefix="step: "),
            )
        ],
        margin=dict(l=0, r=0, t=50, b=0),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    print(f"saved_html={out}")


if __name__ == "__main__":
    main()
