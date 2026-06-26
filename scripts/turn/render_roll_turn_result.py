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


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    return plt, FuncAnimation, FFMpegWriter, PillowWriter, Poly3DCollection


@dataclass
class RenderParams:
    R: float = 0.08
    D: float = 0.40
    L: float = 0.28
    l1: float = 0.18
    l3: float = 0.22
    l5: float = 0.14
    body_dims: tuple[float, float, float] = (0.24, 0.20, 0.10)
    motor_r: float = 0.03
    motor_l: float = 0.04
    wheel_width: float = 0.04
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


def draw_box(ax, center: np.ndarray, size: tuple[float, float, float], rot: np.ndarray, color: str, poly_cls):
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
    verts = [center + rot @ c for c in corners]
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[4], verts[7], verts[3], verts[0]],
    ]
    ax.add_collection3d(poly_cls(faces, facecolors=color, edgecolors="k", linewidths=0.5, alpha=1.0))


def draw_cylinder(ax, center, radius, width, rot, color, spin_angle=None, resolution=24):
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    y = np.linspace(-width / 2.0, width / 2.0, 2)
    theta_grid, y_grid = np.meshgrid(theta, y)

    x_grid = radius * np.cos(theta_grid)
    z_grid = radius * np.sin(theta_grid)
    if spin_angle is not None:
        cs, ss = np.cos(spin_angle), np.sin(spin_angle)
        x_grid, z_grid = x_grid * cs - z_grid * ss, x_grid * ss + z_grid * cs

    X = np.zeros_like(x_grid)
    Y = np.zeros_like(y_grid)
    Z = np.zeros_like(z_grid)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            point = center + rot @ np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]])
            X[i, j], Y[i, j], Z[i, j] = point

    ax.plot_surface(X, Y, Z, color=color, alpha=1.0, shade=False)

    if spin_angle is not None:
        for ang in [0.0, np.pi / 2.0]:
            lx = radius * np.cos(spin_angle + ang)
            lz = radius * np.sin(spin_angle + ang)
            p1 = center + rot @ np.array([lx, -width / 2.0, lz])
            p2 = center + rot @ np.array([lx, width / 2.0, lz])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "w-", lw=1.0)


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


def load_result(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    yaw_refs = np.asarray(data["yaw_refs"], dtype=float).reshape(-1) if "yaw_refs" in data.files else np.zeros(len(states))
    roll_refs = np.asarray(data["roll_refs"], dtype=float).reshape(-1) if "roll_refs" in data.files else np.zeros(len(states))
    if len(yaw_refs) < len(states):
        yaw_refs = np.pad(yaw_refs, (0, len(states) - len(yaw_refs)), mode="edge")
    if len(roll_refs) < len(states):
        roll_refs = np.pad(roll_refs, (0, len(states) - len(roll_refs)), mode="edge")
    meta = {
        "dt": float(scalar(data, "dt", 0.005)),
        "final_reason": str(scalar(data, "final_reason", "unknown")),
        "controller": str(scalar(data, "controller", "unknown")),
        "target_deg": scalar(data, "target_deg", np.nan),
        "v_ref": scalar(data, "v_ref", np.nan),
    }
    return states, actions, yaw_refs, roll_refs, meta


def reconstruct_global_xy(states: np.ndarray, dt: float):
    gx = np.zeros(len(states), dtype=float)
    gy = np.zeros(len(states), dtype=float)
    for i in range(1, len(states)):
        v = states[i - 1, 3]
        yaw = states[i - 1, 6]
        gx[i] = gx[i - 1] + v * np.cos(yaw) * dt
        gy[i] = gy[i - 1] + v * np.sin(yaw) * dt
    return gx, gy


def output_fps(dt: float, stride: int, speed: float) -> int:
    return int(np.clip(round(speed / max(dt * stride, 1e-6)), 8, 60))


class BalanceTurnRollRenderer:
    def __init__(self, states, actions, yaw_refs, roll_refs, meta, params, poly_cls):
        self.states = states
        self.actions = actions
        self.yaw_refs = yaw_refs
        self.roll_refs = roll_refs
        self.meta = meta
        self.p = params
        self.poly_cls = poly_cls
        self.gx, self.gy = reconstruct_global_xy(states, params.dt)

        self.plt, *_ = _require_matplotlib()
        self.fig = self.plt.figure(figsize=(14, 8))
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax_info = self.fig.add_subplot(122)
        self.wheel_spin_l = 0.0
        self.wheel_spin_r = 0.0

    def draw_five_bar_leg(self, body_center, rot_body, rz, wheel_axis, yaw, roll, side, idx):
        p = self.p
        wheel_center = wheel_axis + rz @ np.array([0.0, side * p.D / 2.0, 0.0])
        if idx > 0:
            v = self.states[idx - 1, 3]
            yaw_rate = self.states[idx - 1, 7]
            self.wheel_spin_l -= (v - yaw_rate * p.D / 2.0) / p.R * p.dt
            self.wheel_spin_r -= (v + yaw_rate * p.D / 2.0) / p.R * p.dt
        spin = self.wheel_spin_l if side == 1 else self.wheel_spin_r
        wheel_rot = rotation_matrices(yaw=yaw, roll=roll, pitch=0.0)
        draw_cylinder(self.ax3d, wheel_center, p.R, p.wheel_width, wheel_rot, "#111111", spin_angle=spin)

        dy_body = p.body_dims[1] / 2.0
        dx_motor = p.l5 / 2.0
        leg_y = side * (dy_body + p.motor_l)
        motor_centers = []
        for dx_signed in [dx_motor, -dx_motor]:
            motor_local = np.array([dx_signed, side * (dy_body + p.motor_l / 2.0), 0.0], dtype=float)
            motor_center = body_center + rot_body @ motor_local
            motor_centers.append(motor_center)
            draw_cylinder(self.ax3d, motor_center, p.motor_r, p.motor_l, rot_body, "#222222", resolution=16)

        vec_local = rot_body.T @ (wheel_center - body_center)
        foot_x, foot_z = vec_local[0], vec_local[2]

        knee_a_x, knee_a_z = solve_two_link_ik(foot_x - dx_motor, foot_z, p.l1, p.l3, knee_dir=1.0)
        knee_e_x, knee_e_z = solve_two_link_ik(foot_x + dx_motor, foot_z, p.l1, p.l3, knee_dir=-1.0)

        a_tip = body_center + rot_body @ np.array([dx_motor, leg_y, 0.0])
        knee_a = body_center + rot_body @ np.array([dx_motor + knee_a_x, leg_y, knee_a_z])
        e_tip = body_center + rot_body @ np.array([-dx_motor, leg_y, 0.0])
        knee_e = body_center + rot_body @ np.array([-dx_motor + knee_e_x, leg_y, knee_e_z])
        # End the visual leg exactly at the wheel center.  The two-link IK is
        # solved in the side leg plane, then the final visual segment can slant
        # slightly outboard to the wheel center.
        foot_tip = wheel_center

        for p1, p2 in [(a_tip, knee_a), (knee_a, foot_tip), (e_tip, knee_e), (knee_e, foot_tip)]:
            self.ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k-", lw=2.2)

        self.ax3d.plot(
            [motor_centers[0][0], motor_centers[1][0]],
            [motor_centers[0][1], motor_centers[1][1]],
            [motor_centers[0][2], motor_centers[1][2]],
            color="black",
            lw=1.3,
            alpha=0.8,
        )

    def draw_robot(self, idx: int):
        p = self.p
        theta, _theta_dot, _x, x_dot, phi, _phi_dot, yaw, _yaw_dot, roll, _roll_dot = self.states[idx, :10]
        rot_body = rotation_matrices(yaw=yaw, roll=roll, pitch=phi)
        rz = rotation_matrices(yaw=yaw, roll=0.0, pitch=0.0)
        wheel_axis = np.array([self.gx[idx], self.gy[idx], p.R], dtype=float)
        hip_offset = rz @ np.array([p.L * np.sin(theta), 0.0, p.L * np.cos(theta)], dtype=float)
        body_center = wheel_axis + hip_offset

        draw_box(self.ax3d, body_center, p.body_dims, rot_body, "#dddddd", self.poly_cls)
        for side in [1, -1]:
            self.draw_five_bar_leg(body_center, rot_body, rz, wheel_axis, yaw, roll, side, idx)

        yaw_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self.ax3d.quiver(
            body_center[0],
            body_center[1],
            body_center[2] + 0.12,
            0.35 * yaw_dir[0],
            0.35 * yaw_dir[1],
            0.0,
            linewidth=2.0,
            arrow_length_ratio=0.2,
            color="tab:blue",
        )

        left_top = body_center + rot_body @ np.array([0.0, +p.body_dims[1] / 2.0, +p.body_dims[2] / 2.0])
        right_top = body_center + rot_body @ np.array([0.0, -p.body_dims[1] / 2.0, +p.body_dims[2] / 2.0])
        self.ax3d.plot([left_top[0], right_top[0]], [left_top[1], right_top[1]], [left_top[2], right_top[2]], color="tab:red", lw=2.0)

    def draw_info_panel(self, idx: int):
        self.ax_info.clear()
        self.ax_info.set_title("Balance + Turn + Roll Result", fontsize=12)
        self.ax_info.axis("off")

        theta, theta_dot, x, x_dot, phi, phi_dot, yaw, yaw_dot, roll, roll_dot = self.states[idx, :10]
        if idx < len(self.actions):
            action = self.actions[idx]
        else:
            action = self.actions[-1]
        action = np.pad(action, (0, max(0, 6 - len(action))), mode="constant")
        T, Tp, T_yaw, F_roll = action[:4]
        yaw_ref = self.yaw_refs[min(idx, len(self.yaw_refs) - 1)]
        roll_ref = self.roll_refs[min(idx, len(self.roll_refs) - 1)]

        lines = [
            f"Frame / Step: {idx} / {len(self.states) - 1}",
            f"Time: {idx * self.p.dt:.3f} s",
            f"Controller: {self.meta.get('controller', 'unknown')}",
            f"Final reason = {self.meta.get('final_reason', 'unknown')}",
            "",
            "State:",
            f"theta      = {theta:+.4f} rad  ({np.rad2deg(theta):+.2f} deg)",
            f"theta_dot  = {theta_dot:+.4f} rad/s",
            f"x          = {x:+.4f} m",
            f"x_dot/v    = {x_dot:+.4f} m/s",
            f"phi        = {phi:+.4f} rad  ({np.rad2deg(phi):+.2f} deg)",
            f"phi_dot    = {phi_dot:+.4f} rad/s",
            f"yaw/delta  = {yaw:+.4f} rad  ({np.rad2deg(yaw):+.2f} deg)",
            f"yaw_ref    = {yaw_ref:+.4f} rad  ({np.rad2deg(yaw_ref):+.2f} deg)",
            f"yaw_dot    = {yaw_dot:+.4f} rad/s",
            f"roll/psi   = {roll:+.4f} rad  ({np.rad2deg(roll):+.2f} deg)",
            f"roll_ref   = {roll_ref:+.4f} rad  ({np.rad2deg(roll_ref):+.2f} deg)",
            f"roll_dot   = {roll_dot:+.4f} rad/s",
            "",
            "Control action:",
            f"T          = {T:+.4f}",
            f"Tp         = {Tp:+.4f}",
            f"T_yaw      = {T_yaw:+.4f}",
            f"F_roll     = {F_roll:+.4f}",
            "",
            "Black links = closed-chain five-bar visual links",
            "Red bar = body roll visual reference",
        ]
        self.ax_info.text(0.02, 0.98, "\n".join(lines), transform=self.ax_info.transAxes, va="top", ha="left", fontsize=9, family="monospace")

        inset = self.ax_info.inset_axes([0.08, 0.04, 0.86, 0.28])
        t = np.arange(len(self.states)) * self.p.dt
        inset.plot(t, np.rad2deg(self.states[:, 6]), label="yaw deg")
        inset.plot(t[: len(self.yaw_refs)], np.rad2deg(self.yaw_refs[: len(t)]), "--", label="yaw_ref deg")
        inset.plot(t, np.rad2deg(self.states[:, 8]), label="roll deg")
        inset.plot(t, np.rad2deg(self.states[:, 0]), label="theta deg")
        inset.axvline(idx * self.p.dt, color="k", lw=1.0, alpha=0.5)
        inset.set_xlabel("time / s")
        inset.grid(True, alpha=0.3)
        inset.legend(fontsize=8, loc="best")

    def update(self, idx: int):
        self.ax3d.clear()
        cx, cy = self.gx[idx], self.gy[idx]
        view_range = 0.8
        self.ax3d.set_xlim(cx - view_range, cx + view_range)
        self.ax3d.set_ylim(cy - view_range, cy + view_range)
        self.ax3d.set_zlim(0.0, 0.8)
        self.ax3d.set_xlabel("World X / m")
        self.ax3d.set_ylabel("World Y / m")
        self.ax3d.set_zlabel("Z / m")
        self.ax3d.set_title("Balance + Turning + Roll Five-Bar Animation")
        self.ax3d.set_proj_type("ortho")
        self.ax3d.set_box_aspect((1.0, 1.0, 0.75))
        self.ax3d.view_init(elev=24, azim=-58)

        xx, yy = np.meshgrid(np.linspace(cx - view_range, cx + view_range, 8), np.linspace(cy - view_range, cy + view_range, 8))
        self.ax3d.plot_wireframe(xx, yy, np.zeros_like(xx), color="gray", alpha=0.25, linewidth=0.5)
        self.ax3d.plot(self.gx[: idx + 1], self.gy[: idx + 1], np.zeros(idx + 1), "b-", lw=2.0)
        self.draw_robot(idx)
        self.draw_info_panel(idx)
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--gif", default=None)
    parser.add_argument("--title", default=None, help="Kept for CLI compatibility; 3D renderer uses a fixed title.")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()

    if args.out is None and args.gif is None:
        raise ValueError("Specify at least one of --out or --gif.")

    plt, FuncAnimation, FFMpegWriter, PillowWriter, Poly3DCollection = _require_matplotlib()
    states, actions, yaw_refs, roll_refs, meta = load_result(args.npz)
    params = RenderParams(dt=float(meta["dt"]))
    fps = args.fps if args.fps else output_fps(params.dt, max(1, args.stride), args.speed)

    renderer = BalanceTurnRollRenderer(states, actions, yaw_refs, roll_refs, meta, params, Poly3DCollection)
    frames = list(range(0, len(states), max(1, args.stride)))
    if frames[-1] != len(states) - 1:
        frames.append(len(states) - 1)

    anim = FuncAnimation(renderer.fig, renderer.update, frames=frames, interval=1000 / fps, blit=False)
    print(f"states={len(states)} frames={len(frames)} fps={fps}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(out), writer=FFMpegWriter(fps=fps, bitrate=1800))
        print(f"saved_mp4={out}")
    if args.gif:
        gif = Path(args.gif)
        gif.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(gif), writer=PillowWriter(fps=fps))
        print(f"saved_gif={gif}")
    plt.close(renderer.fig)


if __name__ == "__main__":
    main()
