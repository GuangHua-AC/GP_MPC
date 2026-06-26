from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from terrain.render_terrain_result import (
    RenderParams,
    _require_matplotlib,
    draw_box,
    draw_cylinder,
    output_fps,
    rotation_matrices,
    scalar,
    solve_two_link_ik,
)
from wheel_legged.dynamics.parameters import WheelLeggedParams
from wheel_legged.dynamics.terrain import terrain_heights


def load_result(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    meta = {
        "dt": float(scalar(data, "dt", 0.005)),
        "controller": str(scalar(data, "controller", "unknown")),
        "final_reason": str(scalar(data, "final_reason", "unknown")),
        "terrain_mode": str(scalar(data, "terrain_mode", "panorama")),
        "phase_names": [str(x) for x in np.asarray(data["phase_names"]).reshape(-1)],
    }
    return data, meta


def array_or(data, key: str, default):
    if key not in data.files:
        return np.asarray(default)
    return np.asarray(data[key])


class PanoramaRenderer:
    def __init__(self, data, meta, params, terrain_params, poly_cls):
        self.data = data
        self.states = np.asarray(data["states"], dtype=float)
        self.actions = np.asarray(data["actions"], dtype=float)
        self.phase_ids = np.asarray(data["phase_ids"], dtype=int)
        self.world_x = np.asarray(array_or(data, "world_x", self.states[:, 2]), dtype=float)
        self.world_y = np.asarray(array_or(data, "world_y", np.zeros(len(self.states))), dtype=float)
        self.l0_refs = np.asarray(array_or(data, "l0_refs", np.full(len(self.states), 0.32)), dtype=float)
        self.yaw_refs = np.asarray(array_or(data, "yaw_refs", np.zeros(len(self.states))), dtype=float)
        self.support_rolls = np.asarray(array_or(data, "support_rolls", np.zeros(len(self.states))), dtype=float)
        self.terrain_diffs = np.asarray(array_or(data, "terrain_diffs", np.zeros(len(self.states))), dtype=float)
        self.leg_diff_refs = np.asarray(array_or(data, "leg_diff_refs", np.zeros(len(self.states))), dtype=float)
        self.external_forces = np.asarray(array_or(data, "external_forces", np.zeros(len(self.states))), dtype=float)
        self.meta = meta
        self.p = params
        self.tp = terrain_params
        self.poly_cls = poly_cls
        self.plt, *_ = _require_matplotlib()
        self.fig = self.plt.figure(figsize=(17, 9))
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax_info = self.fig.add_subplot(222)
        self.ax_plot = self.fig.add_subplot(224)
        self.wheel_spin = 0.0
        self.x_min = min(-0.55, float(np.min(self.world_x)) - 0.55)
        self.x_max = max(6.60, float(np.max(self.world_x)) + 0.85)
        self.y_min = min(-1.15, float(np.min(self.world_y)) - 0.75)
        self.y_max = max(1.15, float(np.max(self.world_y)) + 0.75)
        self.obstacle_center = np.asarray(array_or(data, "obstacle_center", np.array([1.40, 0.0, 0.22])), dtype=float).reshape(3)
        self.obstacle_size = np.asarray(array_or(data, "obstacle_size", np.array([0.36, 0.32, 0.44])), dtype=float).reshape(3)
        self.height_column_center = np.asarray(array_or(data, "height_column_center", np.array([1.92, 0.275, 0.09])), dtype=float).reshape(3)
        self.height_column_size = np.asarray(array_or(data, "height_column_size", np.array([0.16, 0.16, 0.18])), dtype=float).reshape(3)
        self.height_table_top_center = np.asarray(array_or(data, "height_table_top_center", np.array([2.40, 0.275, 0.44])), dtype=float).reshape(3)
        self.height_table_top_size = np.asarray(array_or(data, "height_table_top_size", np.array([0.46, 0.58, 0.035])), dtype=float).reshape(3)
        self.height_table_leg_size = np.asarray(array_or(data, "height_table_leg_size", np.array([0.035, 0.035, 0.42])), dtype=float).reshape(3)

    def ground_at(self, x: float) -> tuple[float, float]:
        return terrain_heights(float(x), self.meta["terrain_mode"], self.tp)

    def draw_five_bar_leg(self, body_center, rot_body, wheel_center, side, idx):
        p = self.p
        if idx > 0:
            self.wheel_spin -= self.states[idx - 1, 3] / p.R * p.dt
        draw_cylinder(self.ax3d, wheel_center, p.R, p.wheel_width, rot_body, "#111111", spin_angle=self.wheel_spin)
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
        foot_tip = wheel_center
        for p1, p2 in [(a_tip, knee_a), (knee_a, foot_tip), (e_tip, knee_e), (knee_e, foot_tip)]:
            self.ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k-", lw=2.0)
        self.ax3d.plot(
            [motor_centers[0][0], motor_centers[1][0]],
            [motor_centers[0][1], motor_centers[1][1]],
            [motor_centers[0][2], motor_centers[1][2]],
            color="black",
            lw=1.1,
            alpha=0.8,
        )

    def draw_robot(self, idx: int):
        p = self.p
        theta, _theta_dot, path_s, _x_dot, phi, _phi_dot, yaw, _yaw_dot, roll, _roll_dot, alpha = self.states[idx, :11]
        l0 = 0.30 + 0.10 * np.sin(alpha)
        left_h, right_h = self.ground_at(path_s)
        avg_h = 0.5 * (left_h + right_h)
        rot_body = rotation_matrices(yaw, roll, phi)
        centerline = np.array([self.world_x[idx], self.world_y[idx], avg_h + p.R], dtype=float)
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=float)
        body_center = centerline + p.R * np.sin(theta) * forward + np.array([0.0, 0.0, l0], dtype=float)
        draw_box(self.ax3d, body_center, p.body_dims, rot_body, "#dddddd", self.poly_cls)
        left_wheel = np.array([self.world_x[idx], self.world_y[idx], left_h + p.R], dtype=float) + lateral * (p.D / 2.0)
        right_wheel = np.array([self.world_x[idx], self.world_y[idx], right_h + p.R], dtype=float) - lateral * (p.D / 2.0)
        self.draw_five_bar_leg(body_center, rot_body, left_wheel, +1, idx)
        self.draw_five_bar_leg(body_center, rot_body, right_wheel, -1, idx)
        arrow = rot_body @ np.array([0.38, 0.0, 0.0])
        self.ax3d.quiver(body_center[0], body_center[1], body_center[2] + 0.13, arrow[0], arrow[1], 0.0, linewidth=2.0, arrow_length_ratio=0.2, color="tab:blue")
        left_top = body_center + rot_body @ np.array([0.0, +p.body_dims[1] / 2.0, +p.body_dims[2] / 2.0])
        right_top = body_center + rot_body @ np.array([0.0, -p.body_dims[1] / 2.0, +p.body_dims[2] / 2.0])
        self.ax3d.plot([left_top[0], right_top[0]], [left_top[1], right_top[1]], [left_top[2], right_top[2]], color="tab:red", lw=2.0)

    def phase_ranges(self):
        ranges = []
        for phase_id, name in enumerate(self.meta["phase_names"]):
            idxs = np.where(self.phase_ids == phase_id)[0]
            if len(idxs) == 0:
                continue
            ranges.append((idxs, name))
        return ranges

    def draw_map(self):
        yaws = self.states[:, 6]
        lateral = np.column_stack([-np.sin(yaws), np.cos(yaws)])
        left_xy = np.column_stack([self.world_x, self.world_y]) + lateral * (self.p.D / 2.0)
        right_xy = np.column_stack([self.world_x, self.world_y]) - lateral * (self.p.D / 2.0)
        left_z = np.asarray([self.ground_at(s)[0] for s in self.states[:, 2]])
        right_z = np.asarray([self.ground_at(s)[1] for s in self.states[:, 2]])
        self.ax3d.plot(left_xy[:, 0], left_xy[:, 1], left_z, color="tab:green", lw=3.0)
        self.ax3d.plot(right_xy[:, 0], right_xy[:, 1], right_z, color="sienna", lw=3.0)
        self.ax3d.plot(self.world_x, self.world_y, np.zeros_like(self.world_x), color="0.60", lw=1.0, alpha=0.8)
        draw_box(self.ax3d, self.obstacle_center, tuple(self.obstacle_size), np.eye(3), "#b84a4a", self.poly_cls)
        self.ax3d.text(
            self.obstacle_center[0],
            self.obstacle_center[1],
            self.obstacle_center[2] + self.obstacle_size[2] * 0.65,
            "large obstacle",
            ha="center",
            fontsize=8,
            color="#7a1f1f",
        )
        draw_box(self.ax3d, self.height_column_center, tuple(self.height_column_size), np.eye(3), "#4f8cc9", self.poly_cls)
        self.ax3d.text(
            self.height_column_center[0],
            self.height_column_center[1],
            self.height_column_center[2] + self.height_column_size[2] * 0.75,
            "raise",
            ha="center",
            fontsize=8,
            color="#1f4f7a",
        )
        draw_box(self.ax3d, self.height_table_top_center, tuple(self.height_table_top_size), np.eye(3), "#8a6d3b", self.poly_cls)
        table_top = self.height_table_top_center
        table_size = self.height_table_top_size
        leg_size = self.height_table_leg_size
        for sx in [-1.0, 1.0]:
            for sy in [-1.0, 1.0]:
                leg_center = np.array(
                    [
                        table_top[0] + sx * (table_size[0] / 2.0 - leg_size[0] / 2.0),
                        table_top[1] + sy * (table_size[1] / 2.0 - leg_size[1] / 2.0),
                        leg_size[2] / 2.0,
                    ],
                    dtype=float,
                )
                draw_box(self.ax3d, leg_center, tuple(leg_size), np.eye(3), "#8a6d3b", self.poly_cls)
        self.ax3d.text(
            table_top[0],
            table_top[1],
            table_top[2] + 0.08,
            "lower",
            ha="center",
            fontsize=8,
            color="#5a421e",
        )
        colors = ["#e8f2ff", "#fff2d8", "#e9f7e7", "#fde9e5"]
        for i, (idxs, name) in enumerate(self.phase_ranges()):
            self.ax3d.plot(self.world_x[idxs], self.world_y[idxs], np.full(len(idxs), 0.012), color=colors[i % len(colors)], lw=8.0, alpha=0.8)
            label = {
                "balance_walk": "Balance Walk",
                "obstacle_turn": "Turn Around Obstacle",
                "height": "Height",
                "adaptive_terrain": "Adaptive Terrain",
            }.get(name, name)
            mid = idxs[len(idxs) // 2]
            self.ax3d.text(self.world_x[mid], self.world_y[mid] - 0.28, 0.045, label, ha="center", va="bottom", fontsize=8)

    def draw_info(self, idx: int):
        self.ax_info.clear()
        s = self.states[idx]
        phase = self.meta["phase_names"][int(self.phase_ids[idx])]
        action = self.actions[min(idx, len(self.actions) - 1)]
        lines = [
            "Panorama Showcase",
            f"Frame: {idx} / {len(self.states) - 1}",
            f"Time: {idx * self.p.dt:.2f} s",
            f"Phase: {phase}",
            f"Final reason: {self.meta['final_reason']}",
            "",
            f"path_s  = {s[2]:+.3f} m",
            f"world   = ({self.world_x[idx]:+.2f}, {self.world_y[idx]:+.2f}) m",
            f"yaw     = {np.rad2deg(s[6]):+.2f} deg",
            f"yaw_ref = {np.rad2deg(self.yaw_refs[idx]):+.2f} deg",
            f"L0_ref  = {self.l0_refs[idx]:+.3f} m",
            f"theta   = {s[0]:+.4f} rad",
            f"phi     = {s[4]:+.4f} rad",
            f"roll    = {np.rad2deg(s[8]):+.2f} deg",
            f"terr_d  = {self.terrain_diffs[idx]:+.4f} m",
            f"leg_d   = {s[12]:+.4f} m",
            f"push_x  = {self.external_forces[idx]:+.1f} N",
            "",
            f"T       = {action[0]:+.3f}",
            f"Tp      = {action[1]:+.3f}",
            f"Fheight = {action[4]:+.3f}",
            f"leg_cmd = {action[5]:+.3f}",
        ]
        self.ax_info.axis("off")
        self.ax_info.text(0.02, 0.98, "\n".join(lines), transform=self.ax_info.transAxes, va="top", ha="left", fontsize=9, family="monospace")

    def draw_plot(self, idx: int):
        self.ax_plot.clear()
        t = np.arange(len(self.states)) * self.p.dt
        self.ax_plot.plot(t, np.rad2deg(self.states[:, 8]), label="roll deg")
        self.ax_plot.plot(t, self.states[:, 12] * 100.0, label="leg diff cm")
        self.ax_plot.plot(t, self.terrain_diffs * 100.0, label="terrain diff cm")
        self.ax_plot.plot(t, self.l0_refs, label="L0 ref m")
        self.ax_plot.axvline(idx * self.p.dt, color="k", lw=1.0, alpha=0.5)
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.set_xlabel("time / s")
        self.ax_plot.legend(fontsize=8, loc="best")

    def update(self, idx: int):
        self.ax3d.clear()
        self.ax3d.set_title("Full Panorama Map: Balance + Roll Turn + Height + Adaptive Terrain")
        self.ax3d.set_xlim(self.x_min, self.x_max)
        self.ax3d.set_ylim(self.y_min, self.y_max)
        self.ax3d.set_zlim(0.0, 0.85)
        self.ax3d.set_xlabel("World X / m")
        self.ax3d.set_ylabel("World Y / m")
        self.ax3d.set_zlabel("Z / m")
        self.ax3d.set_proj_type("ortho")
        self.ax3d.set_box_aspect((2.15, 1.0, 0.72))
        self.ax3d.view_init(elev=25, azim=-45)
        xx, yy = np.meshgrid(np.linspace(self.x_min, self.x_max, 18), np.linspace(self.y_min, self.y_max, 10))
        self.ax3d.plot_wireframe(xx, yy, np.zeros_like(xx), color="gray", alpha=0.18, linewidth=0.5)
        self.draw_map()
        self.ax3d.plot(self.world_x[: idx + 1], self.world_y[: idx + 1], np.zeros(idx + 1), color="0.25", lw=1.4, alpha=0.75)
        self.draw_robot(idx)
        self.draw_info(idx)
        self.draw_plot(idx)
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="outputs/panorama/results/panorama_showcase_adaptive_pd.npz")
    parser.add_argument("--out", default="outputs/panorama/videos/panorama_showcase.mp4")
    parser.add_argument("--gif", default=None)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()

    plt, FuncAnimation, FFMpegWriter, PillowWriter, Poly3DCollection = _require_matplotlib()
    data, meta = load_result(args.npz)
    params = RenderParams(dt=float(meta["dt"]))
    terrain_params = WheelLeggedParams()
    fps = args.fps if args.fps else output_fps(params.dt, max(1, args.stride), args.speed)
    renderer = PanoramaRenderer(data, meta, params, terrain_params, Poly3DCollection)
    frames = list(range(0, len(renderer.states), max(1, args.stride)))
    if frames[-1] != len(renderer.states) - 1:
        frames.append(len(renderer.states) - 1)
    anim = FuncAnimation(renderer.fig, renderer.update, frames=frames, interval=1000 / fps, blit=False)
    print(f"states={len(renderer.states)} frames={len(frames)} fps={fps}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(out), writer=FFMpegWriter(fps=fps, bitrate=2200))
        print(f"saved_mp4={out}")
    if args.gif:
        gif = Path(args.gif)
        gif.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(gif), writer=PillowWriter(fps=fps))
        print(f"saved_gif={gif}")
    plt.close(renderer.fig)


if __name__ == "__main__":
    main()
