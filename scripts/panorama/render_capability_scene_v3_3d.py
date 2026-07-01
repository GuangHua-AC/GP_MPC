from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from terrain.render_terrain_result import (
    RenderParams,
    _require_matplotlib,
    draw_box,
    draw_cylinder,
    output_fps,
    rotation_matrices,
    solve_two_link_ik,
)


ZONE_LABELS = ["Balance", "Turn", "Height", "Known Terrain", "Adaptive Terrain", "Jump"]
ROUTE = "Balance -> Turn -> Height -> Known Terrain -> Adaptive Terrain -> Jump"


def scalar(scene: dict[str, np.ndarray], key: str, default=None):
    if key not in scene:
        return default
    arr = np.asarray(scene[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_scene(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def terrain_at(scene: dict[str, np.ndarray], x: np.ndarray | float) -> np.ndarray:
    profile = np.asarray(scene["ground_profile"] if "ground_profile" in scene else scene["terrain_profile"], dtype=float)
    return np.interp(x, profile[:, 0], profile[:, 1])


def action_text(scene: dict[str, np.ndarray], idx: int) -> str:
    return (
        f"t = {float(scene['time'][idx]):05.2f} s\n"
        f"zone = {str(scene['zone_name'][idx])}\n"
        f"action_norm = {float(scene['action_norm'][idx]):.2f}\n"
        f"source = {str(scene['source_status'][idx])}"
    )


class CapabilityScene3DRenderer:
    def __init__(self, scene: dict[str, np.ndarray], poly_cls, style: str, theme: str, show_metrics: bool, camera: str, debug_clearance: bool = False):
        self.scene = scene
        self.poly_cls = poly_cls
        self.style = style
        self.theme = theme
        self.show_metrics = show_metrics
        self.camera = camera
        self.debug_clearance = debug_clearance
        self.is_story = "scene_variant" in scene and "story" in str(scene["scene_variant"].reshape(-1)[0])
        self.p = RenderParams(
            R=0.10,
            D=0.48,
            l1=0.22,
            l3=0.27,
            l5=0.16,
            body_dims=(0.34, 0.26, 0.14),
            motor_r=0.035,
            motor_l=0.05,
            wheel_width=0.055,
        )
        self.plt, *_ = _require_matplotlib()
        self.fig = self.plt.figure(figsize=(16, 9))
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.wheel_spin = 0.0
        self.n = len(scene["time"])
        self.duration = float(scene["time"][-1] + 1.0 / float(scalar(scene, "fps", 30)))

        self.current_idx = 0
        if theme == "clear":
            self.bg = "#ffffff"
            self.text = "#0b1220"
            self.muted = "#1f2937"
            self.ground = "#f4efe4"
            self.grid = "#d9e1ea"
            self.robot = "#fffdf7"
            self.axis_color = "#e5e7eb"
            self.surface_alpha = 0.16
            self.wire_alpha = 0.14
            self.object_alpha = 1.0
            self.robot_lw = 2.35
        elif style == "dark":
            self.bg = "#111827"
            self.text = "#f9fafb"
            self.muted = "#cbd5e1"
            self.ground = "#475569"
            self.grid = "#94a3b8"
            self.robot = "#e5e7eb"
            self.axis_color = "#64748b"
            self.surface_alpha = 0.72
            self.wire_alpha = 0.22
            self.object_alpha = 1.0
            self.robot_lw = 1.9
        else:
            self.bg = "#ffffff"
            self.text = "#111827"
            self.muted = "#475569"
            self.ground = "#d8d2c7"
            self.grid = "#94a3b8"
            self.robot = "#dddddd"
            self.axis_color = "#d1d5db"
            self.surface_alpha = 0.72
            self.wire_alpha = 0.22
            self.object_alpha = 1.0
            self.robot_lw = 1.9

    def camera_bounds(self, idx: int) -> tuple[tuple[float, float], tuple[float, float], bool]:
        intro = int(2.0 / self.duration * self.n)
        outro = int(2.0 / self.duration * self.n)
        wide = self.camera == "wide" or idx < intro or idx >= self.n - outro
        if wide:
            return (-0.8, 30.8), (-1.35, 1.35), True
        x = float(self.scene["camera_x"][idx] if "camera_x" in self.scene else self.scene["scene_x"][idx])
        y = float(self.scene["scene_y"][idx])
        return (max(-0.8, x - 2.25), min(30.8, x + 2.25)), (y - 1.05, y + 1.05), False

    def draw_ground(self, xlim: tuple[float, float], ylim: tuple[float, float]):
        xs = np.linspace(xlim[0], xlim[1], 80)
        ys = np.linspace(ylim[0], ylim[1], 18)
        xx, yy = np.meshgrid(xs, ys)
        zz = terrain_at(self.scene, xx)
        if self.theme == "clear":
            active = int(self.scene["zone_id"][self.current_idx])
            zx0, zx1 = active * 5.0, active * 5.0 + 5.0
            mx = (xs >= max(xlim[0], zx0)) & (xs <= min(xlim[1], zx1))
            if np.any(mx):
                cxx, cyy = np.meshgrid(xs[mx], ys)
                czz = terrain_at(self.scene, cxx)
                self.ax3d.plot_surface(cxx, cyy, czz, color="#fff3c4", alpha=0.42, linewidth=0, shade=False)
            self.ax3d.plot_wireframe(xx, yy, zz + 0.003, color=self.grid, alpha=self.wire_alpha, linewidth=0.32)
        else:
            self.ax3d.plot_surface(xx, yy, zz, color=self.ground, alpha=self.surface_alpha, linewidth=0, shade=False)
            self.ax3d.plot_wireframe(xx, yy, zz + 0.003, color=self.grid, alpha=self.wire_alpha, linewidth=0.45)
        centerline_z = terrain_at(self.scene, xs)
        self.ax3d.plot(xs, np.zeros_like(xs), centerline_z + 0.012, color="#111827" if self.theme == "clear" else "#4b5563", lw=1.65 if self.theme == "clear" else 1.3)

        # Zone center labels.
        for center, label in zip([2.5, 7.5, 12.5, 17.5, 22.5, 27.5], ZONE_LABELS):
            if xlim[0] - 0.2 <= center <= xlim[1] + 0.2:
                self.ax3d.text(center, -0.92, 0.06, label, ha="center", fontsize=8, color=self.muted)

        # Trajectory trace from all scene data.
        x = np.asarray(self.scene["scene_x"], dtype=float)
        y = np.asarray(self.scene["scene_y"], dtype=float)
        z = terrain_at(self.scene, x) + 0.02
        mask = (x >= xlim[0] - 0.5) & (x <= xlim[1] + 0.5)
        self.ax3d.plot(x[mask], y[mask], z[mask], color="#2563eb", lw=1.6 if self.theme == "clear" else 1.2, alpha=0.68 if self.theme == "clear" else 0.45)

    def draw_objects(self, xlim: tuple[float, float]):
        eye = np.eye(3)
        if self.is_story:
            objects = [
                (np.array([7.25, 0.0, terrain_at(self.scene, 7.25) + 0.23]), (0.60, 0.48, 0.46), "#d95f4f", "obstacle\nturn required"),
                (np.array([11.55, 0.45, terrain_at(self.scene, 11.55) + 0.11]), (0.60, 0.34, 0.22), "#d6a85d", "raise body\nobstacle clearance"),
                (np.array([13.55, 0.0, 0.66]), (1.05, 0.76, 0.08), "#8a6d3b", "lower body\nlow clearance"),
                (np.array([16.95, 0.50, terrain_at(self.scene, 16.95) + 0.08]), (1.00, 0.32, 0.16), "#8bbf88", "known terrain"),
                (np.array([26.85, 0.0, -0.32]), (1.05, 1.12, 0.40), "#64748b", "gap"),
            ]
        else:
            objects = [
                (np.array([11.2, 0.42, terrain_at(self.scene, 11.2) + 0.09]), (0.56, 0.28, 0.18), "#d6a85d", "low"),
                (np.array([12.0, 0.42, terrain_at(self.scene, 12.0) + 0.22]), (0.52, 0.28, 0.44), "#d6a85d", "raise"),
                (np.array([13.45, 0.42, terrain_at(self.scene, 13.45) + 0.07]), (0.78, 0.42, 0.14), "#8a6d3b", "lower"),
                (np.array([16.95, 0.50, terrain_at(self.scene, 16.95) + 0.08]), (1.25, 0.32, 0.16), "#8bbf88", "known terrain"),
                (np.array([26.85, 0.0, -0.32]), (1.05, 1.12, 0.40), "#64748b", "gap"),
            ]
        for center, size, color, label in objects:
            if xlim[0] - 0.6 <= center[0] <= xlim[1] + 0.6:
                draw_box(self.ax3d, center, size, eye, color, self.poly_cls)
                if label != "gap":
                    self.ax3d.text(center[0], center[1] - 0.36, center[2] + size[2] * 0.62, label, ha="center", fontsize=8 if self.theme == "clear" else 7, color=self.text)

        # Turn arc, adaptive feedback, jump trajectory hints.
        if xlim[0] < 7.5 < xlim[1]:
            a = np.linspace(-0.7, 1.05, 80)
            if self.is_story:
                bypass_x = np.linspace(5.0, 10.0, 110)
                tau = (bypass_x - 5.0) / 5.0
                bypass_y = 0.82 * np.sin(np.pi * tau) ** 2
                self.ax3d.plot(bypass_x, bypass_y, terrain_at(self.scene, bypass_x) + 0.08, color="#1d4ed8", lw=2.8 if self.theme == "clear" else 2.1, ls="--")
                self.ax3d.text(7.25, -0.56, 0.58, "front obstacle\nturn required", ha="center", fontsize=9 if self.theme == "clear" else 8, color="#b91c1c")
            else:
                self.ax3d.plot(7.5 + 1.15 * np.cos(a), 0.15 + 0.58 * np.sin(a), 0.10 + 0 * a, color="#1d4ed8", lw=2.6 if self.theme == "clear" else 2.0, ls="--")
                self.ax3d.text(7.5, -0.55, 0.35, "yaw_ref=30deg", ha="center", fontsize=9 if self.theme == "clear" else 8, color="#1d4ed8")
        if xlim[0] < 22.5 < xlim[1]:
            self.ax3d.text(22.5, -0.55, 0.42, "blind feedback", ha="center", fontsize=9 if self.theme == "clear" else 8, color="#b45309")
        if xlim[0] < 27.0 < xlim[1]:
            jx = np.linspace(25.7, 28.2, 90)
            jz = terrain_at(self.scene, jx) + 0.28 + 0.62 * np.sin((jx - jx[0]) / (jx[-1] - jx[0]) * np.pi)
            self.ax3d.plot(jx, np.zeros_like(jx), jz, color="#7c3aed", lw=2.8 if self.theme == "clear" else 2.0, ls="--")
            self.ax3d.text(27.0, -0.55, 0.76, "Jump / smoke tests / exploratory", ha="center", fontsize=9 if self.theme == "clear" else 8, color="#6d28d9")

    def draw_five_bar_leg(self, body_center: np.ndarray, rot_body: np.ndarray, wheel_center: np.ndarray, side: float, idx: int):
        p = self.p
        if idx > 0:
            dx = float(self.scene["scene_x"][idx] - self.scene["scene_x"][idx - 1])
            self.wheel_spin -= dx / max(p.R, 1e-6)
        draw_cylinder(self.ax3d, wheel_center, p.R, p.wheel_width, rot_body, "#050505", spin_angle=self.wheel_spin, resolution=18)

        dy_body = p.body_dims[1] / 2.0
        dx_motor = p.l5 / 2.0
        leg_y = side * (dy_body + p.motor_l)
        motor_centers = []
        for dx_signed in [dx_motor, -dx_motor]:
            motor_local = np.array([dx_signed, side * (dy_body + p.motor_l / 2.0), 0.0], dtype=float)
            motor_center = body_center + rot_body @ motor_local
            motor_centers.append(motor_center)
            draw_cylinder(self.ax3d, motor_center, p.motor_r, p.motor_l, rot_body, "#050505", resolution=14)

        vec_local = rot_body.T @ (wheel_center - body_center)
        foot_x, foot_z = vec_local[0], vec_local[2]
        knee_a_x, knee_a_z = solve_two_link_ik(foot_x - dx_motor, foot_z, p.l1, p.l3, knee_dir=1.0)
        knee_e_x, knee_e_z = solve_two_link_ik(foot_x + dx_motor, foot_z, p.l1, p.l3, knee_dir=-1.0)
        a_tip = body_center + rot_body @ np.array([dx_motor, leg_y, 0.0])
        knee_a = body_center + rot_body @ np.array([dx_motor + knee_a_x, leg_y, knee_a_z])
        e_tip = body_center + rot_body @ np.array([-dx_motor, leg_y, 0.0])
        knee_e = body_center + rot_body @ np.array([-dx_motor + knee_e_x, leg_y, knee_e_z])

        for p1, p2 in [(a_tip, knee_a), (knee_a, wheel_center), (e_tip, knee_e), (knee_e, wheel_center)]:
            self.ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#000000", lw=self.robot_lw)
        self.ax3d.plot(
            [motor_centers[0][0], motor_centers[1][0]],
            [motor_centers[0][1], motor_centers[1][1]],
            [motor_centers[0][2], motor_centers[1][2]],
            color="black",
            lw=1.25 if self.theme == "clear" else 1.0,
            alpha=0.95 if self.theme == "clear" else 0.8,
        )

    def draw_wheel_legged_five_link_robot(self, idx: int):
        p = self.p
        x = float(self.scene["scene_x"][idx])
        y = float(self.scene["scene_y"][idx])
        pitch = float(np.clip(self.scene["robot_phi"][idx], -0.4, 0.4))
        yaw = float(self.scene["robot_yaw"][idx])
        roll = float(np.clip(self.scene["robot_roll"][idx], -0.45, 0.45))
        l0 = float(np.clip(self.scene["robot_L0"][idx], 0.22, 0.42))
        leg_diff = float(np.clip(self.scene["robot_leg_diff"][idx], -0.09, 0.09))
        jump_z = float(np.clip(self.scene["robot_z"][idx], 0.0, 0.65))
        ground = float(terrain_at(self.scene, x))
        rot_body = rotation_matrices(yaw, roll, pitch)
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        body_center = np.array([x, y, ground + p.R + l0 + jump_z], dtype=float)
        draw_box(self.ax3d, body_center, p.body_dims, rot_body, self.robot, self.poly_cls)

        left_wheel = np.array([x, y, float(terrain_at(self.scene, x)) + p.R + jump_z + 0.5 * leg_diff], dtype=float) + lateral * (p.D / 2.0)
        right_wheel = np.array([x, y, float(terrain_at(self.scene, x)) + p.R + jump_z - 0.5 * leg_diff], dtype=float) - lateral * (p.D / 2.0)
        self.draw_five_bar_leg(body_center, rot_body, left_wheel, +1.0, idx)
        self.draw_five_bar_leg(body_center, rot_body, right_wheel, -1.0, idx)

        arrow = rot_body @ np.array([0.36, 0.0, 0.0])
        self.ax3d.quiver(body_center[0], body_center[1], body_center[2] + 0.11, arrow[0], arrow[1], 0.0, linewidth=2.8 if self.theme == "clear" else 2.0, arrow_length_ratio=0.22, color="#1d4ed8")

    def draw_overlay(self, idx: int):
        zone = str(self.scene["zone_name"][idx])
        metric = str(self.scene["metric_texts"][idx])
        self.ax3d.text2D(0.5, 0.965, "Wheel-Legged Robot Capability Scene", transform=self.ax3d.transAxes, ha="center", fontsize=19 if self.theme == "clear" else 18, weight="bold", color=self.text)
        self.ax3d.text2D(0.5, 0.925, "wheel_legged_new / GP-PMPC + task showcase", transform=self.ax3d.transAxes, ha="center", fontsize=10.5 if self.theme == "clear" else 10, color=self.muted)
        if self.show_metrics:
            self.ax3d.text2D(0.02, 0.86, metric, transform=self.ax3d.transAxes, ha="left", va="top", fontsize=9 if self.theme == "clear" else 8.5, color=self.text)
        self.ax3d.text2D(0.98, 0.86, action_text(self.scene, idx), transform=self.ax3d.transAxes, ha="right", va="top", fontsize=9 if self.theme == "clear" else 8.5, color=self.muted)
        self.ax3d.text2D(0.5, 0.055, ROUTE, transform=self.ax3d.transAxes, ha="center", fontsize=10.5, weight="bold", color=self.text)
        active = int(self.scene["zone_id"][idx])
        progress = "".join("●" if i <= active else "○" for i in range(6))
        self.ax3d.text2D(0.5, 0.025, progress, transform=self.ax3d.transAxes, ha="center", fontsize=13, color="tab:blue")
        self.ax3d.text2D(0.5, 0.89, zone, transform=self.ax3d.transAxes, ha="center", fontsize=13, weight="bold", color=self.text)

    def clearance_warnings(self) -> list[str]:
        warnings: list[str] = []
        x = np.asarray(self.scene["scene_x"], dtype=float)
        y = np.asarray(self.scene["scene_y"], dtype=float)
        yaw = np.unwrap(np.asarray(self.scene["robot_yaw"], dtype=float))
        l0 = np.asarray(self.scene["robot_L0"], dtype=float)
        zone = np.asarray(self.scene["source_task"]).astype(str)

        turn = zone == "turn"
        if np.any(turn):
            in_turn_block_x = (x >= 6.95) & (x <= 7.55)
            in_turn_block_y = (y >= -0.24) & (y <= 0.24)
            bad = np.where(turn & in_turn_block_x & in_turn_block_y)[0]
            if len(bad):
                warnings.append(f"turn_path_intersects_obstacle frame={int(bad[0])}")

        jumps = np.rad2deg(np.abs(np.diff(yaw)))
        if len(jumps) and float(np.max(jumps)) > 45.0:
            warnings.append(f"heading_jump max_deg={float(np.max(jumps)):.2f}")

        height = zone == "height"
        if np.any(height):
            beam_x = (x >= 13.05) & (x <= 14.10)
            beam_y = (y >= -0.38) & (y <= 0.38)
            body_top = terrain_at(self.scene, x) + self.p.R + l0 + self.p.body_dims[2] / 2.0
            bad = np.where(height & beam_x & beam_y & (body_top > 0.62))[0]
            if len(bad):
                warnings.append(f"height_down_body_beam_overlap frame={int(bad[0])} body_top={float(body_top[bad[0]]):.3f}")

            up_x = (x >= 11.25) & (x <= 11.85)
            up_y = (y >= 0.28) & (y <= 0.62)
            bad = np.where(height & up_x & up_y)[0]
            if len(bad):
                warnings.append(f"height_up_path_intersects_side_obstacle frame={int(bad[0])}")

        if self.debug_clearance:
            if warnings:
                for warning in warnings:
                    print(f"warning_clearance={warning}")
            else:
                print("clearance_check=OK")
        return warnings

    def update(self, idx: int):
        self.ax3d.clear()
        self.fig.patch.set_facecolor(self.bg)
        self.ax3d.set_facecolor(self.bg)
        xlim, ylim, wide = self.camera_bounds(idx)
        self.ax3d.set_xlim(*xlim)
        self.ax3d.set_ylim(*ylim)
        self.ax3d.set_zlim(-0.35, 1.05)
        self.ax3d.set_xlabel("World X / m")
        self.ax3d.set_ylabel("World Y / m")
        self.ax3d.set_zlabel("Z / m")
        if self.theme == "clear":
            for axis in (self.ax3d.xaxis, self.ax3d.yaxis, self.ax3d.zaxis):
                axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                axis._axinfo["grid"]["color"] = (0.88, 0.91, 0.95, 0.45)
                axis._axinfo["grid"]["linewidth"] = 0.55
            self.ax3d.tick_params(colors="#111827", labelsize=9)
        self.ax3d.set_proj_type("ortho")
        self.ax3d.set_box_aspect((2.45, 1.0, 0.72))
        self.ax3d.view_init(elev=24, azim=-48 if not wide else -52)
        self.current_idx = idx
        self.draw_ground(xlim, ylim)
        self.draw_objects(xlim)
        if int(self.scene["event_flags"][idx]) & 1:
            x = float(self.scene["scene_x"][idx])
            y = float(self.scene["scene_y"][idx])
            z = float(terrain_at(self.scene, x)) + 0.75
            self.ax3d.quiver(x - 0.8, y - 0.25, z + 0.15, 0.55, 0.2, -0.35, color="#dc2626", linewidth=3.2 if self.theme == "clear" else 2.5, arrow_length_ratio=0.24)
            self.ax3d.text(x - 0.98, y - 0.35, z + 0.33, "30N push", color="#dc2626", fontsize=9 if self.theme == "clear" else 8)
        self.draw_wheel_legged_five_link_robot(idx)
        self.draw_overlay(idx)
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/panorama/capability_scene/capability_scene_v3_3d.npz")
    parser.add_argument("--out-mp4", default="outputs/panorama/videos/capability_scene_v3_3d.mp4")
    parser.add_argument("--out-gif", default="outputs/panorama/videos/capability_scene_v3_3d.gif")
    parser.add_argument("--snapshot", default="outputs/panorama/figures/capability_scene_v3_3d_snapshot.png")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--style", choices=["thesis", "dark"], default="thesis")
    parser.add_argument("--theme", choices=["default", "clear"], default="default")
    parser.add_argument("--camera", choices=["auto", "wide"], default="auto")
    parser.add_argument("--debug-clearance", action="store_true")
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.theme == "clear":
        if args.out_mp4 == "outputs/panorama/videos/capability_scene_v3_3d.mp4":
            args.out_mp4 = "outputs/panorama/videos/capability_scene_v3_3d_clear.mp4"
        if args.out_gif == "outputs/panorama/videos/capability_scene_v3_3d.gif":
            args.out_gif = "outputs/panorama/videos/capability_scene_v3_3d_clear.gif"
        if args.snapshot == "outputs/panorama/figures/capability_scene_v3_3d_snapshot.png":
            args.snapshot = "outputs/panorama/figures/capability_scene_v3_3d_clear_snapshot.png"

    plt, FuncAnimation, FFMpegWriter, PillowWriter, Poly3DCollection = _require_matplotlib()
    scene = load_scene(args.input)
    source_fps = int(scalar(scene, "fps", 30))
    fps = args.fps if args.fps else output_fps(1.0 / source_fps, max(1, args.stride), args.speed)
    renderer = CapabilityScene3DRenderer(
        scene,
        Poly3DCollection,
        args.style,
        args.theme,
        args.show_metrics,
        "wide" if args.camera == "wide" else "auto",
        debug_clearance=args.debug_clearance,
    )
    renderer.clearance_warnings()
    frames = list(range(0, len(scene["time"]), max(1, args.stride)))
    if frames[-1] != len(scene["time"]) - 1:
        frames.append(len(scene["time"]) - 1)

    snapshot = Path(args.snapshot)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    renderer.update(min(len(scene["time"]) - 1, int(0.55 * len(scene["time"]))))
    renderer.fig.savefig(snapshot, dpi=args.dpi, bbox_inches="tight", facecolor=renderer.fig.get_facecolor())
    print(f"saved_snapshot={snapshot}")

    print(f"states={len(scene['time'])} frames={len(frames)} fps={fps} video_duration≈{len(frames) / fps:.2f}s sim_duration≈{float(scene['time'][-1] + 1/source_fps):.2f}s")
    if args.out_mp4:
        anim = FuncAnimation(renderer.fig, renderer.update, frames=frames, interval=1000 / fps, blit=False)
        out_mp4 = Path(args.out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(out_mp4), writer=FFMpegWriter(fps=fps, bitrate=2600), dpi=args.dpi)
        print(f"saved_mp4={out_mp4}")
    if args.out_gif:
        out_gif = Path(args.out_gif)
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        gif_stride = 2
        gif_frames = frames[::gif_stride]
        if gif_frames[-1] != frames[-1]:
            gif_frames.append(frames[-1])
        gif_anim = FuncAnimation(renderer.fig, renderer.update, frames=gif_frames, interval=1000 / min(fps, 12), blit=False)
        gif_anim.save(str(out_gif), writer=PillowWriter(fps=min(fps, 12)), dpi=max(90, int(args.dpi * 0.62)))
        print(f"saved_gif={out_gif}")
    plt.close(renderer.fig)


if __name__ == "__main__":
    main()
