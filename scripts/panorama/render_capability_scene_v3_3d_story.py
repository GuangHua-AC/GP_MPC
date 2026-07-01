from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from terrain.render_terrain_result import _require_matplotlib, draw_box, output_fps, rotation_matrices, solve_two_link_ik

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from render_capability_scene_v3_3d import CapabilityScene3DRenderer, load_scene, scalar, terrain_at  # noqa: E402


def draw_box_alpha(ax, center: np.ndarray, size: tuple[float, float, float], rot: np.ndarray, color: str, poly_cls, alpha: float):
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
    ax.add_collection3d(poly_cls(faces, facecolors=color, edgecolors="#166534", linewidths=0.7, alpha=alpha))


class StoryLayoutRenderer(CapabilityScene3DRenderer):
    def is_motion(self) -> bool:
        return "motion" in str(self.scene.get("scene_variant", np.asarray([""])).reshape(-1)[0])

    def is_known_fix(self) -> bool:
        return "known_terrain_fix" in str(self.scene.get("scene_variant", np.asarray([""])).reshape(-1)[0])

    def draw_objects(self, xlim: tuple[float, float]):
        eye = np.eye(3)
        layout = "layout" in str(self.scene.get("scene_variant", np.asarray([""])).reshape(-1)[0])
        motion = self.is_motion()
        known_fix = self.is_known_fix()
        if not layout and not motion:
            return super().draw_objects(xlim)

        if motion:
            known_center = np.array([17.05, 0.255, 0.045]) if known_fix else np.array([17.05, 0.24, 0.09])
            known_size = (1.35, 0.075, 0.09) if known_fix else (1.35, 0.24, 0.18)
            known_label = "Known Terrain\none wheel on narrow rail" if known_fix else "Known Terrain\nleft wheel on block"
            objects = [
                (np.array([7.25, 0.0, terrain_at(self.scene, 7.25) + 0.23]), (0.60, 0.48, 0.46), "#d95f4f", "front obstacle\nturn required"),
                (np.array([11.45, 0.50, terrain_at(self.scene, 11.45) + 0.045]), (0.28, 0.18, 0.09), "#d6a85d", "raise body\nsmall curb"),
                (np.array([13.55, 0.0, 0.54]), (1.04, 0.78, 0.07), "#8a6d3b", "lower body\npass under low clearance"),
                (known_center, known_size, "#8bbf88", known_label),
                (np.array([26.85, 0.0, -0.32]), (1.05, 1.12, 0.40), "#64748b", "gap"),
            ]
        else:
            objects = [
                (np.array([7.25, 0.0, terrain_at(self.scene, 7.25) + 0.23]), (0.60, 0.48, 0.46), "#d95f4f", "front obstacle\nturn required"),
                (np.array([11.45, 0.45, terrain_at(self.scene, 11.45) + 0.10]), (0.58, 0.32, 0.20), "#d6a85d", "raise body\nobstacle clearance"),
                (np.array([13.55, 0.0, 0.66]), (1.08, 0.76, 0.08), "#8a6d3b", "lower body\nlow clearance"),
                (np.array([17.05, 0.0, terrain_at(self.scene, 17.05) + 0.08]), (1.25, 0.52, 0.16), "#8bbf88", "Known Terrain\nknown Δh"),
                (np.array([26.85, 0.0, -0.32]), (1.05, 1.12, 0.40), "#64748b", "gap"),
            ]
        for center, size, color, label in objects:
            if xlim[0] - 0.6 <= center[0] <= xlim[1] + 0.6:
                if known_fix and label.startswith("Known Terrain"):
                    draw_box_alpha(self.ax3d, center, size, eye, color, self.poly_cls, alpha=0.58)
                else:
                    draw_box(self.ax3d, center, size, eye, color, self.poly_cls)
                if label != "gap":
                    self.ax3d.text(center[0], center[1] - 0.42, center[2] + size[2] * 0.68, label, ha="center", fontsize=8, color=self.text)

        if motion and xlim[0] - 0.6 <= 13.55 <= xlim[1] + 0.6:
            for leg_x in (13.08, 14.02):
                for leg_y in (-0.34, 0.34):
                    draw_box(
                        self.ax3d,
                        np.array([leg_x, leg_y, 0.25]),
                        (0.055, 0.055, 0.50),
                        eye,
                        "#7a5a2a",
                        self.poly_cls,
                    )

        if xlim[0] < 7.5 < xlim[1]:
            bypass_x = np.linspace(5.0, 10.0, 110)
            tau = (bypass_x - 5.0) / 5.0
            bypass_y = 0.82 * np.sin(np.pi * tau) ** 2
            self.ax3d.plot(bypass_x, bypass_y, terrain_at(self.scene, bypass_x) + 0.08, color="#1d4ed8", lw=2.8 if self.theme == "clear" else 2.1, ls="--")

        if xlim[0] < 12.5 < xlim[1]:
            up_z0 = 0.18 if motion else 0.18
            up_len = 0.46 if motion else 0.32
            self.ax3d.text(11.45, 0.80, 0.56 if motion else 0.46, "L0 up\nraised body\nclearance", ha="center", fontsize=8, color="#1d4ed8")
            self.ax3d.quiver(11.45, 0.70, up_z0, 0.0, 0.0, up_len, color="#1d4ed8", linewidth=3.0 if motion else 2.2, arrow_length_ratio=0.22)
            self.ax3d.text(13.55, -0.58, 0.45 if motion else 0.54, "L0 down\nlower body\npass under", ha="center", fontsize=8, color="#1d4ed8")
            self.ax3d.quiver(13.55, -0.48, 0.75, 0.0, 0.0, -0.42 if motion else -0.30, color="#1d4ed8", linewidth=3.0 if motion else 2.2, arrow_length_ratio=0.22)

        if xlim[0] < 17.5 < xlim[1]:
            dh = 0.12 if known_fix else 0.18
            self.ax3d.quiver(16.45, -0.42, 0.02, 0.0, 0.0, dh, color="#15803d", linewidth=2.6, arrow_length_ratio=0.24)
            self.ax3d.text(16.55, -0.54, 0.22 if known_fix else 0.27, "known dH\nterrain prior", ha="center", fontsize=8, color="#15803d")
            self.ax3d.text(17.55, 0.54 if known_fix else 0.60, 0.31 if known_fix else 0.34, "leg_diff tracking\nbalance on uneven terrain", ha="center", fontsize=8, color="#15803d")

        if xlim[0] < 22.5 < xlim[1]:
            self.ax3d.text(22.5, -0.58, 0.40, "no terrain prior\nblind feedback\nadaptive leg_diff", ha="center", fontsize=8, color="#b45309")

        if xlim[0] < 27.0 < xlim[1]:
            jx = np.linspace(25.7, 28.2, 90)
            jz = terrain_at(self.scene, jx) + 0.28 + 0.62 * np.sin((jx - jx[0]) / (jx[-1] - jx[0]) * np.pi)
            self.ax3d.plot(jx, np.zeros_like(jx), jz, color="#7c3aed", lw=2.8 if self.theme == "clear" else 2.0, ls="--")
            self.ax3d.text(27.0, -0.55, 0.76, "Jump / smoke tests / exploratory", ha="center", fontsize=9, color="#6d28d9")

    def draw_wheel_legged_five_link_robot(self, idx: int):
        if not self.is_motion() or str(self.scene["source_task"][idx]) != "known_terrain":
            return super().draw_wheel_legged_five_link_robot(idx)

        p = self.p
        x = float(self.scene["scene_x"][idx])
        y = float(self.scene["scene_y"][idx])
        pitch = float(np.clip(self.scene["robot_phi"][idx], -0.4, 0.4))
        yaw = float(self.scene["robot_yaw"][idx])
        roll = 0.0
        l0 = float(np.clip(self.scene["robot_L0"][idx], 0.25, 0.40))
        jump_z = float(np.clip(self.scene["robot_z"][idx], 0.0, 0.65))
        ground = float(terrain_at(self.scene, x))
        block_active = 16.35 <= x <= 17.85
        high_side = (0.12 if self.is_known_fix() else 0.18) if block_active else 0.0
        rot_body = rotation_matrices(yaw, roll, pitch)
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        body_center = np.array([x, y, ground + p.R + l0 + jump_z + 0.02], dtype=float)
        draw_box(self.ax3d, body_center, p.body_dims, rot_body, self.robot, self.poly_cls)

        left_wheel = np.array([x, y, ground + p.R + jump_z + high_side], dtype=float) + lateral * (p.D / 2.0)
        right_wheel = np.array([x, y, ground + p.R + jump_z], dtype=float) - lateral * (p.D / 2.0)
        self.draw_five_bar_leg(body_center, rot_body, left_wheel, +1.0, idx)
        self.draw_five_bar_leg(body_center, rot_body, right_wheel, -1.0, idx)
        if self.is_known_fix() and block_active:
            self.redraw_known_near_leg(body_center, rot_body, left_wheel, +1.0)
        arrow = rot_body @ np.array([0.36, 0.0, 0.0])
        self.ax3d.quiver(body_center[0], body_center[1], body_center[2] + 0.11, arrow[0], arrow[1], 0.0, linewidth=2.8, arrow_length_ratio=0.22, color="#1d4ed8")

    def redraw_known_near_leg(self, body_center: np.ndarray, rot_body: np.ndarray, wheel_center: np.ndarray, side: float):
        p = self.p
        dy_body = p.body_dims[1] / 2.0
        dx_motor = p.l5 / 2.0
        leg_y = side * (dy_body + p.motor_l)
        vec_local = rot_body.T @ (wheel_center - body_center)
        foot_x, foot_z = vec_local[0], vec_local[2]
        knee_a_x, knee_a_z = solve_two_link_ik(foot_x - dx_motor, foot_z, p.l1, p.l3, knee_dir=1.0)
        knee_e_x, knee_e_z = solve_two_link_ik(foot_x + dx_motor, foot_z, p.l1, p.l3, knee_dir=-1.0)
        a_tip = body_center + rot_body @ np.array([dx_motor, leg_y, 0.0])
        knee_a = body_center + rot_body @ np.array([dx_motor + knee_a_x, leg_y, knee_a_z])
        e_tip = body_center + rot_body @ np.array([-dx_motor, leg_y, 0.0])
        knee_e = body_center + rot_body @ np.array([-dx_motor + knee_e_x, leg_y, knee_e_z])
        for p1, p2 in [(a_tip, knee_a), (knee_a, wheel_center), (e_tip, knee_e), (knee_e, wheel_center)]:
            self.ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#000000", lw=3.1, alpha=1.0)
        theta = np.linspace(0.0, 2.0 * np.pi, 64)
        forward = rot_body @ np.array([1.0, 0.0, 0.0])
        vertical = rot_body @ np.array([0.0, 0.0, 1.0])
        circle = wheel_center[:, None] + p.R * (np.outer(forward, np.cos(theta)) + np.outer(vertical, np.sin(theta)))
        self.ax3d.plot(circle[0], circle[1], circle[2], color="#000000", lw=2.5, alpha=1.0)

    def draw_overlay(self, idx: int):
        super().draw_overlay(idx)
        if self.is_known_fix():
            note = "Closeout-data visualization: task npz mapped to one 3D scene; obstacles explain context, not live autonomous avoidance."
            self.ax3d.text2D(0.5, 0.087, note, transform=self.ax3d.transAxes, ha="center", fontsize=7.4, color=self.muted)

    def clearance_warnings(self) -> list[str]:
        debug = self.debug_clearance
        self.debug_clearance = False
        warnings = super().clearance_warnings()
        self.debug_clearance = debug

        if self.is_known_fix():
            p = self.p
            block_center_y = 0.255
            block_width_y = 0.075
            block_height = 0.09
            block_x_min, block_x_max = 16.375, 17.725
            block_y_min = block_center_y - block_width_y / 2.0
            block_y_max = block_center_y + block_width_y / 2.0
            body_half_y = p.body_dims[1] / 2.0
            if block_width_y >= p.D / 2.0:
                warnings.append("known_terrain_block_too_wide_for_single_wheel_lane")
            if abs(block_center_y) - block_width_y / 2.0 <= body_half_y:
                warnings.append("known_terrain_block_overlaps_body_center_lane")
            if block_height > 0.14:
                warnings.append(f"known_terrain_block_too_tall height={block_height:.3f}")

            zone = np.asarray(self.scene["source_task"]).astype(str)
            x = np.asarray(self.scene["scene_x"], dtype=float)
            y = np.asarray(self.scene["scene_y"], dtype=float)
            active = (zone == "known_terrain") & (x >= block_x_min) & (x <= block_x_max)
            if np.any(active):
                body_y_min = y[active] - body_half_y
                body_y_max = y[active] + body_half_y
                overlaps_body_y = (body_y_max > block_y_min) & (body_y_min < block_y_max)
                if np.any(overlaps_body_y):
                    bad = np.where(active)[0][np.where(overlaps_body_y)[0][0]]
                    warnings.append(f"known_terrain_body_y_overlap frame={int(bad)}")

                yaw = np.asarray(self.scene["robot_yaw"], dtype=float)[active]
                left_y = y[active] + np.cos(yaw) * p.D / 2.0
                if np.any((left_y < block_y_min - 0.03) | (left_y > block_y_max + 0.03)):
                    bad = np.where(active)[0][0]
                    warnings.append(f"known_terrain_left_wheel_not_on_side_rail frame={int(bad)}")
                if block_width_y > 0.09:
                    warnings.append("known_terrain_near_leg_projection_may_be_occluded")

        if debug:
            if warnings:
                for warning in warnings:
                    print(f"warning_clearance={warning}")
            else:
                print("clearance_check=OK")
        return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/panorama/capability_scene/capability_scene_v3_3d_story.npz")
    parser.add_argument("--out-mp4", default="outputs/panorama/videos/capability_scene_v3_3d_story.mp4")
    parser.add_argument("--out-gif", default="outputs/panorama/videos/capability_scene_v3_3d_story.gif")
    parser.add_argument("--snapshot", default="outputs/panorama/figures/capability_scene_v3_3d_story_snapshot.png")
    parser.add_argument("--theme", choices=["default", "clear"], default="clear")
    parser.add_argument("--style", choices=["thesis", "dark"], default="thesis")
    parser.add_argument("--camera", choices=["auto", "wide"], default="auto")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--debug-clearance", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    plt, FuncAnimation, FFMpegWriter, PillowWriter, Poly3DCollection = _require_matplotlib()
    scene = load_scene(args.input)
    source_fps = int(scalar(scene, "fps", 30))
    fps = args.fps if args.fps else output_fps(1.0 / source_fps, max(1, args.stride), 1.0)
    renderer = StoryLayoutRenderer(
        scene,
        Poly3DCollection,
        args.style,
        args.theme,
        True,
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
    print(
        f"states={len(scene['time'])} frames={len(frames)} fps={fps} "
        f"video_duration≈{len(frames) / fps:.2f}s sim_duration≈{float(scene['time'][-1] + 1/source_fps):.2f}s"
    )

    if args.out_mp4:
        anim = FuncAnimation(renderer.fig, renderer.update, frames=frames, interval=1000 / fps, blit=False)
        out_mp4 = Path(args.out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(out_mp4), writer=FFMpegWriter(fps=fps, bitrate=2600), dpi=args.dpi)
        print(f"saved_mp4={out_mp4}")
    if args.out_gif and not args.no_gif:
        out_gif = Path(args.out_gif)
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        gif_frames = frames[::2]
        if gif_frames[-1] != frames[-1]:
            gif_frames.append(frames[-1])
        gif_anim = FuncAnimation(renderer.fig, renderer.update, frames=gif_frames, interval=1000 / min(fps, 12), blit=False)
        gif_anim.save(str(out_gif), writer=PillowWriter(fps=min(fps, 12)), dpi=max(90, int(args.dpi * 0.62)))
        print(f"saved_gif={out_gif}")
    plt.close(renderer.fig)


if __name__ == "__main__":
    main()
