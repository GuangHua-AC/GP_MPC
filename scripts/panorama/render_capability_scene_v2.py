from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt, FuncAnimation, FFMpegWriter, PillowWriter, Circle, FancyArrowPatch, Polygon, Rectangle


PALETTES = {
    "thesis": {
        "bg": "#ffffff",
        "text": "#111827",
        "muted": "#475569",
        "ground": "#d8d2c7",
        "ground_edge": "#475569",
        "panel": "#f8fafc",
        "panel_edge": "#cbd5e1",
        "blue": "#1d4ed8",
        "red": "#dc2626",
        "green": "#15803d",
        "orange": "#b45309",
        "purple": "#7c3aed",
        "robot": "#f8fafc",
        "wheel": "#111827",
        "water": "#64748b",
    },
    "clean": {
        "bg": "#fbfdff",
        "text": "#0f172a",
        "muted": "#64748b",
        "ground": "#d6d3d1",
        "ground_edge": "#334155",
        "panel": "#ffffff",
        "panel_edge": "#94a3b8",
        "blue": "#2563eb",
        "red": "#b91c1c",
        "green": "#047857",
        "orange": "#c2410c",
        "purple": "#6d28d9",
        "robot": "#ffffff",
        "wheel": "#020617",
        "water": "#64748b",
    },
    "dark": {
        "bg": "#111827",
        "text": "#f9fafb",
        "muted": "#cbd5e1",
        "ground": "#475569",
        "ground_edge": "#94a3b8",
        "panel": "#1f2937",
        "panel_edge": "#64748b",
        "blue": "#38bdf8",
        "red": "#f87171",
        "green": "#22c55e",
        "orange": "#fb923c",
        "purple": "#c084fc",
        "robot": "#e5e7eb",
        "wheel": "#030712",
        "water": "#0f172a",
    },
}


ZONE_LABELS = ["Balance", "Turn", "Height", "Known Terrain", "Adaptive Terrain", "Jump"]
ROUTE = "Balance -> Turn -> Height -> Known Terrain -> Adaptive Terrain -> Jump"


def load_scene(path: str):
    data = np.load(path, allow_pickle=True)
    return {k: np.asarray(data[k]) for k in data.files}


def terrain_at(scene, x: np.ndarray | float) -> np.ndarray:
    profile = np.asarray(scene["terrain_profile"], dtype=float)
    return np.interp(x, profile[:, 0], profile[:, 1])


def rotated_rect(cx: float, cz: float, w: float, h: float, angle: float) -> np.ndarray:
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], dtype=float)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.array([cx, cz])


def draw_robot(ax, scene, idx: int, colors, alpha: float = 1.0):
    _, _, _, _, Circle, _, Polygon, _ = require_matplotlib()
    x = float(scene["scene_x"][idx])
    z_ground = float(terrain_at(scene, x))
    phi = float(scene["robot_phi"][idx])
    roll = float(scene["robot_roll"][idx])
    l0 = float(np.clip(scene["robot_L0"][idx], 0.22, 0.44))
    leg_diff = float(np.clip(scene["robot_leg_diff"][idx], -0.09, 0.09))
    extra_z = float(scene["robot_z"][idx])
    wheel_r = 0.12
    wheel_dx = 0.30
    body_w = 0.62
    body_h = 0.28
    left_x = x - wheel_dx
    right_x = x + wheel_dx
    left_z = float(terrain_at(scene, left_x)) + wheel_r + 0.5 * leg_diff + extra_z
    right_z = float(terrain_at(scene, right_x)) + wheel_r - 0.5 * leg_diff + extra_z
    body_z = z_ground + wheel_r + l0 + extra_z
    body_angle = float(np.clip(phi + 0.35 * roll, -0.45, 0.45))

    for wx, wz in [(left_x, left_z), (right_x, right_z)]:
        ax.add_patch(Circle((wx, wz), wheel_r, facecolor="#ffffff", edgecolor=colors["wheel"], lw=1.4, alpha=alpha, zorder=24))
    body_poly = rotated_rect(x, body_z, body_w, body_h, body_angle)
    ax.add_patch(Polygon(body_poly, closed=True, facecolor=colors["robot"], edgecolor=colors["wheel"], lw=1.4, alpha=alpha, zorder=25))
    for wx, wz, side in [(left_x, left_z, -1), (right_x, right_z, 1)]:
        hip_x = x + side * 0.21
        hip_z = body_z - 0.12
        knee_x = 0.5 * (hip_x + wx)
        knee_z = max(wz + 0.18, 0.5 * (hip_z + wz) - 0.08)
        ax.plot([hip_x, knee_x, wx], [hip_z, knee_z, wz], color=colors["wheel"], lw=1.6, alpha=alpha, zorder=23)

    yaw = float(scene["robot_yaw"][idx])
    arrow_len = 0.36
    ax.arrow(
        x,
        body_z + 0.24,
        arrow_len * math.cos(yaw),
        0.10 * math.sin(yaw),
        width=0.012,
        head_width=0.07,
        head_length=0.08,
        color=colors["blue"],
        alpha=alpha,
        length_includes_head=True,
        zorder=28,
    )


def draw_static_scene(ax, scene, colors, xlim: tuple[float, float]):
    _, _, _, _, Circle, FancyArrowPatch, Polygon, Rectangle = require_matplotlib()
    profile = np.asarray(scene["terrain_profile"], dtype=float)
    mask = (profile[:, 0] >= xlim[0] - 0.5) & (profile[:, 0] <= xlim[1] + 0.5)
    xs = profile[mask, 0]
    zs = profile[mask, 1]
    ax.fill_between(xs, -0.85, zs, color=colors["ground"], alpha=0.95, zorder=1)
    ax.plot(xs, zs, color=colors["ground_edge"], lw=1.4, zorder=3)

    # Zone boundaries and labels.
    for edge in [5, 10, 15, 20, 25]:
        if xlim[0] - 0.2 <= edge <= xlim[1] + 0.2:
            ax.plot([edge, edge], [-0.5, 2.4], color=colors["muted"], lw=0.8, ls="--", alpha=0.26, zorder=0)
    for center, label in zip([2.5, 7.5, 12.5, 17.5, 22.5, 27.5], ZONE_LABELS):
        if xlim[0] <= center <= xlim[1]:
            ax.text(center, 2.13, label, ha="center", va="bottom", fontsize=10, weight="bold", color=colors["muted"], zorder=20)

    # Balance push marker.
    if xlim[0] < 2.0 < xlim[1]:
        ax.add_patch(FancyArrowPatch((0.95, 1.25), (1.55, 0.72), arrowstyle="-|>", mutation_scale=18, lw=2.2, color=colors["red"], zorder=18))
        ax.text(0.85, 1.38, "30N push", color=colors["red"], fontsize=8, ha="left", zorder=18)

    # Turn arc.
    if xlim[0] < 7.5 < xlim[1]:
        t = np.linspace(-0.7, 1.0, 80)
        ax.plot(7.5 + 1.1 * np.cos(t), 0.70 + 0.45 * np.sin(t), color=colors["blue"], lw=2.0, alpha=0.65, zorder=7)
        ax.text(7.5, 1.42, "yaw_ref=30deg", ha="center", fontsize=8, color=colors["blue"], zorder=18)

    # Height platforms.
    for x, h, label in [(10.9, 0.18, "low"), (11.9, 0.38, "raise"), (13.3, -0.08, "lower")]:
        if xlim[0] < x < xlim[1]:
            base = float(terrain_at(scene, x))
            ax.add_patch(Rectangle((x - 0.42, min(base, base + h)), 0.84, abs(h), facecolor="#d6a85d", edgecolor="#7c4a03", lw=1.0, zorder=8))
            ax.text(x, max(base, base + h) + 0.08, label, ha="center", fontsize=7, color="#7c4a03", zorder=18)

    # Known terrain height difference.
    if xlim[0] < 17.0 < xlim[1]:
        ax.add_patch(FancyArrowPatch((16.4, 1.12), (16.85, 0.30), arrowstyle="<->", mutation_scale=13, lw=1.6, color=colors["green"], zorder=18))
        ax.text(16.55, 1.24, "known Δh", fontsize=8, color=colors["green"], ha="center", zorder=18)

    # Adaptive feedback loop.
    if xlim[0] < 22.4 < xlim[1]:
        ax.add_patch(FancyArrowPatch((21.7, 1.35), (22.28, 0.74), connectionstyle="arc3,rad=-0.4", arrowstyle="-|>", mutation_scale=12, color=colors["orange"], lw=1.6, zorder=18))
        ax.add_patch(FancyArrowPatch((22.38, 0.74), (23.0, 1.35), connectionstyle="arc3,rad=-0.4", arrowstyle="-|>", mutation_scale=12, color=colors["orange"], lw=1.6, zorder=18))
        ax.text(22.35, 1.52, "blind feedback", fontsize=8, color=colors["orange"], ha="center", zorder=18)

    # Jump gap and trajectory guide.
    if xlim[0] < 27.0 < xlim[1]:
        ax.fill_between([26.25, 27.45], [-0.85, -0.85], [-0.55, -0.55], color=colors["water"], alpha=0.55, zorder=2)
        jx = np.linspace(25.65, 28.1, 80)
        jz = 0.45 + 0.85 * np.sin((jx - jx[0]) / (jx[-1] - jx[0]) * np.pi)
        ax.plot(jx, jz, color=colors["purple"], lw=1.8, ls="--", zorder=7)
        ax.text(27.0, 1.55, "Jump / smoke tests / exploratory", ha="center", fontsize=8.5, color=colors["purple"], weight="bold", zorder=18)


class CapabilitySceneRenderer:
    def __init__(self, scene, style: str, show_metrics: bool, camera: str):
        self.scene = scene
        self.colors = PALETTES[style]
        self.show_metrics = show_metrics
        self.camera = camera
        self.plt, *_ = require_matplotlib()
        self.fig, self.ax = self.plt.subplots(figsize=(15.5, 8.2))
        self.fig.patch.set_facecolor(self.colors["bg"])

    def frame_xlim(self, idx: int, wide: bool = False) -> tuple[float, float]:
        if wide or self.camera == "wide":
            return (-0.8, 30.8)
        x = float(self.scene["scene_x"][idx])
        return (max(-0.8, x - 3.0), min(30.8, x + 3.0))

    def draw_progress(self, idx: int):
        _, _, _, _, Circle, *_ = require_matplotlib()
        colors = self.colors
        active = int(self.scene["zone_id"][idx])
        y = -0.70
        x0, x1 = self.ax.get_xlim()
        self.ax.plot([x0 + 0.5, x1 - 0.5], [y, y], color=colors["blue"], lw=2.0, zorder=30)
        centers = np.linspace(x0 + 0.65, x1 - 0.65, 6)
        for i, center in enumerate(centers):
            self.ax.add_patch(Circle((center, y), 0.055, facecolor=colors["blue"] if i <= active else colors["panel"], edgecolor=colors["blue"], lw=1.2, zorder=31))
        self.ax.text((x0 + x1) / 2, y - 0.17, ROUTE, ha="center", va="center", fontsize=9.5, color=colors["text"], weight="bold", zorder=32)

    def draw_overlay(self, idx: int):
        colors = self.colors
        x0, x1 = self.ax.get_xlim()
        self.ax.text((x0 + x1) / 2, 2.55, "Wheel-Legged Robot Capability Scene", ha="center", va="center", fontsize=18, weight="bold", color=colors["text"], zorder=40)
        self.ax.text((x0 + x1) / 2, 2.35, "wheel_legged_new / GP-PMPC + task showcase", ha="center", va="center", fontsize=10, color=colors["muted"], zorder=40)
        metric = str(self.scene["metric_texts"][idx])
        source = str(self.scene["source_status"][idx])
        if self.show_metrics:
            self.ax.text(
                x0 + 0.22,
                2.18,
                metric + f"\nsource={source}",
                ha="left",
                va="top",
                fontsize=8.5,
                color=colors["text"],
                bbox=dict(boxstyle="round,pad=0.34", facecolor=colors["panel"], edgecolor=colors["panel_edge"], alpha=0.94),
                zorder=40,
            )
        t = float(self.scene["time"][idx])
        action = float(self.scene["action_norm"][idx])
        self.ax.text(x1 - 0.22, 2.18, f"t={t:05.2f}s\naction_norm={action:.2f}", ha="right", va="top", fontsize=8.5, color=colors["muted"], zorder=40)

    def update(self, idx: int, wide: bool = False):
        self.ax.clear()
        colors = self.colors
        self.ax.set_facecolor(colors["bg"])
        xlim = self.frame_xlim(idx, wide=wide)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(-0.9, 2.7)
        self.ax.axis("off")
        draw_static_scene(self.ax, self.scene, colors, xlim)

        # Path already traversed.
        upto = slice(0, idx + 1)
        path_x = self.scene["scene_x"][upto]
        path_z = terrain_at(self.scene, path_x) + 0.02
        self.ax.plot(path_x, path_z, color=colors["blue"], lw=1.3, alpha=0.45, zorder=12)

        draw_robot(self.ax, self.scene, idx, colors)
        if int(self.scene["event_flags"][idx]) & 1:
            x = float(self.scene["scene_x"][idx])
            self.ax.annotate("", xy=(x - 0.25, 0.88), xytext=(x - 0.9, 1.25), arrowprops=dict(arrowstyle="-|>", color=colors["red"], lw=3.0), zorder=35)

        self.draw_overlay(idx)
        self.draw_progress(idx)
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/panorama/capability_scene/capability_scene_v2.npz")
    parser.add_argument("--out-mp4", default="outputs/panorama/videos/capability_scene_v2.mp4")
    parser.add_argument("--out-gif", default="outputs/panorama/videos/capability_scene_v2.gif")
    parser.add_argument("--snapshot", default="outputs/panorama/figures/capability_scene_v2_snapshot.png")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--style", choices=["clean", "thesis", "dark"], default="thesis")
    parser.add_argument("--camera", choices=["follow", "wide"], default="follow")
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    plt, FuncAnimation, FFMpegWriter, PillowWriter, *_ = require_matplotlib()
    scene = load_scene(args.input)
    renderer = CapabilitySceneRenderer(scene, args.style, args.show_metrics, args.camera)
    n = len(scene["time"])
    step = max(1, int(round(max(1.0, args.speed))))
    frames = list(range(0, n, step))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    snapshot = Path(args.snapshot)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    renderer.update(min(n - 1, int(0.55 * n)), wide=False)
    renderer.fig.savefig(snapshot, dpi=args.dpi, bbox_inches="tight", facecolor=renderer.fig.get_facecolor())
    print(f"saved_snapshot={snapshot}")

    print(f"states={n} frames={len(frames)} fps={args.fps} video_duration≈{len(frames) / args.fps:.2f}s")

    if args.out_mp4:
        anim = FuncAnimation(renderer.fig, renderer.update, frames=frames, interval=1000 / args.fps, blit=False)
        out_mp4 = Path(args.out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        try:
            anim.save(str(out_mp4), writer=FFMpegWriter(fps=args.fps, bitrate=2400), dpi=args.dpi)
            print(f"saved_mp4={out_mp4}")
        except Exception as exc:  # noqa: BLE001
            print(f"warning_mp4_failed={exc}")

    if args.out_gif:
        out_gif = Path(args.out_gif)
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        gif_stride = max(1, int(round(args.fps / 15)))
        gif_frames = frames[::gif_stride]
        if gif_frames[-1] != frames[-1]:
            gif_frames.append(frames[-1])
        gif_anim = FuncAnimation(renderer.fig, renderer.update, frames=gif_frames, interval=1000 / min(args.fps, 15), blit=False)
        try:
            gif_anim.save(str(out_gif), writer=PillowWriter(fps=min(args.fps, 15)), dpi=max(100, int(args.dpi * 0.72)))
            print(f"saved_gif={out_gif}")
        except Exception as exc:  # noqa: BLE001
            print(f"warning_gif_failed={exc}")
    plt.close(renderer.fig)


if __name__ == "__main__":
    main()
