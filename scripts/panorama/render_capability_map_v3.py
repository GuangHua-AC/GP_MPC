from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from wheel_legged.utils.paths import OUTPUT_DIR

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from render_capability_map_v2 import as_float, format_metric, load_metrics, require_matplotlib  # noqa: E402


ZH_BALANCE = "\u5e73\u8861"
ZH_TURN = "\u8f6c\u5411"
ZH_HEIGHT = "\u53d8\u817f\u9ad8"
ZH_KNOWN_TERRAIN = "\u5df2\u77e5\u5730\u5f62"
ZH_ADAPTIVE_TERRAIN = "\u5730\u5f62\u81ea\u9002\u5e94"
ZH_JUMP = "\u8df3\u8dc3\u63a2\u7d22"
ROADMAP = f"{ZH_BALANCE} -> {ZH_TURN} -> {ZH_HEIGHT} -> {ZH_KNOWN_TERRAIN} -> {ZH_ADAPTIVE_TERRAIN} -> {ZH_JUMP}"


@dataclass(frozen=True)
class VariantStyle:
    name: str
    figsize: tuple[float, float]
    dpi: int
    title_size: float
    zone_title_size: float
    zone_subtitle_size: float
    metric_size: float
    route_size: float
    line_width: float
    robot_scale: float
    contrast: float
    metric_alpha: float
    png_name: str
    pdf_name: str | None


PALETTES = {
    "thesis": {
        "bg": "#ffffff",
        "panel": "#f8fafc",
        "panel_edge": "#cbd5e1",
        "text": "#111827",
        "muted": "#475569",
        "ground": "#d8d2c7",
        "road": "#e2e8f0",
        "blue": "#1d4ed8",
        "green": "#15803d",
        "orange": "#b45309",
        "red": "#dc2626",
        "purple": "#7c3aed",
        "robot": "#f8fafc",
        "wheel": "#111827",
    },
    "defense": {
        "bg": "#fbfdff",
        "panel": "#ffffff",
        "panel_edge": "#64748b",
        "text": "#0f172a",
        "muted": "#334155",
        "ground": "#c9bda9",
        "road": "#cbd5e1",
        "blue": "#0b5fff",
        "green": "#047857",
        "orange": "#c2410c",
        "red": "#b91c1c",
        "purple": "#6d28d9",
        "robot": "#ffffff",
        "wheel": "#020617",
    },
}


VARIANTS = {
    "thesis": VariantStyle(
        name="thesis",
        figsize=(21, 9),
        dpi=260,
        title_size=22,
        zone_title_size=14,
        zone_subtitle_size=8.5,
        metric_size=8,
        route_size=12,
        line_width=1.4,
        robot_scale=1.0,
        contrast=1.0,
        metric_alpha=0.92,
        png_name="wheel_legged_task_big_map_v3_thesis.png",
        pdf_name="wheel_legged_task_big_map_v3_thesis.pdf",
    ),
    "defense": VariantStyle(
        name="defense",
        figsize=(21, 9),
        dpi=260,
        title_size=26,
        zone_title_size=17,
        zone_subtitle_size=10.5,
        metric_size=9.8,
        route_size=15,
        line_width=1.8,
        robot_scale=1.04,
        contrast=1.22,
        metric_alpha=0.98,
        png_name="wheel_legged_task_big_map_v3_defense.png",
        pdf_name="wheel_legged_task_big_map_v3_defense.pdf",
    ),
    "slide": VariantStyle(
        name="defense",
        figsize=(16, 9),
        dpi=220,
        title_size=23,
        zone_title_size=14,
        zone_subtitle_size=8.5,
        metric_size=8,
        route_size=12.5,
        line_width=1.65,
        robot_scale=0.96,
        contrast=1.15,
        metric_alpha=0.98,
        png_name="wheel_legged_task_big_map_v3_slide.png",
        pdf_name=None,
    ),
}


def compact_zone_metrics(metrics: dict[str, dict[str, str] | None]) -> dict[str, list[str]]:
    b = metrics["balance"]
    t = metrics["turn"]
    h = metrics["height"]
    kt = metrics["known_terrain"]
    ta = metrics["terrain_adaptive"]
    j = metrics["jump"]
    return {
        "balance": [
            f"|theta|max {format_metric(as_float(b, 'max_abs_theta'), ' rad')}",
            f"|phi|max {format_metric(as_float(b, 'max_abs_phi'), ' rad')}",
        ],
        "turn": [
            f"yaw err {format_metric(abs(as_float(t, 'final_yaw_error_deg')), ' deg')}",
            f"|roll|max {format_metric(as_float(t, 'max_abs_roll'), ' rad')}",
        ],
        "height": [
            f"L0 RMSE {format_metric(as_float(h, 'rmse_L0_error_after_1s'), ' m')}",
            f"mode {h.get('mode', 'step') if h else 'missing'}",
        ],
        "known_terrain": [
            f"leg err {format_metric(as_float(kt, 'max_abs_leg_diff_error'), ' m')}",
            f"|roll|max {format_metric(as_float(kt, 'max_abs_roll'), ' rad')}",
        ],
        "terrain_adaptive": [
            f"|roll|max {format_metric(as_float(ta, 'max_abs_roll'), ' rad')}",
            f"|leg_diff|max {format_metric(as_float(ta, 'max_abs_leg_diff'), ' m')}",
        ],
        "jump": [
            f"h {format_metric(as_float(j, 'h_target'), ' m')}",
            f"success {j.get('success', 'missing') if j else 'missing'}",
        ],
    }


def draw_robot(ax, x: float, ground_y: float, scale: float, body_height: float, colors, roll: float = 0.0, zorder: int = 10):
    _, _, Circle, _, _, Rectangle = require_matplotlib()
    wheel_r = 0.13 * scale
    wheel_dx = 0.31 * scale
    body_w = 0.72 * scale
    body_h = 0.30 * scale
    wheel_y = ground_y + wheel_r
    body_y = ground_y + body_height * scale
    body = Rectangle(
        (x - body_w / 2, body_y - body_h / 2),
        body_w,
        body_h,
        angle=math.degrees(roll),
        facecolor=colors["robot"],
        edgecolor=colors["wheel"],
        lw=1.25 * scale,
        zorder=zorder,
    )
    ax.add_patch(body)
    for side in (-1, 1):
        wx = x + side * wheel_dx
        ax.add_patch(
            Circle((wx, wheel_y), wheel_r, facecolor="#ffffff", edgecolor=colors["wheel"], lw=1.25 * scale, zorder=zorder)
        )
        hip_x = x + side * wheel_dx * 0.72
        hip_y = body_y - body_h * 0.42
        knee_x = x + side * wheel_dx * 1.05
        knee_y = ground_y + 0.48 * scale
        ax.plot([hip_x, knee_x, wx], [hip_y, knee_y, wheel_y], color=colors["wheel"], lw=1.45 * scale, zorder=zorder)


def draw_metric_box(ax, x: float, lines: list[str], colors, style: VariantStyle):
    text = "\n".join(lines[:2])
    ax.text(
        x,
        6.98,
        text,
        ha="center",
        va="top",
        fontsize=style.metric_size,
        color=colors["text"],
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor=colors["panel"],
            edgecolor=colors["panel_edge"],
            lw=0.7 * style.contrast,
            alpha=style.metric_alpha,
        ),
        zorder=30,
    )


def draw_zone_header(ax, x: float, title: str, subtitle: str, metrics: list[str], colors, style: VariantStyle):
    ax.text(x, 7.82, title, ha="center", va="center", fontsize=style.zone_title_size, weight="bold", color=colors["text"])
    ax.text(x, 7.48, subtitle, ha="center", va="center", fontsize=style.zone_subtitle_size, color=colors["muted"])
    draw_metric_box(ax, x, metrics, colors, style)


def ground_profile(xs: np.ndarray) -> np.ndarray:
    return np.piecewise(
        xs,
        [xs < 4, (xs >= 4) & (xs < 8), (xs >= 8) & (xs < 12), (xs >= 12) & (xs < 16), (xs >= 16) & (xs < 20), xs >= 20],
        [
            1.02,
            lambda x: 1.02 + 0.14 * np.sin((x - 4) / 4 * np.pi),
            lambda x: 1.00 + 0.22 * ((x > 9.25) & (x < 10.15)) - 0.14 * ((x > 10.7) & (x < 11.6)),
            lambda x: 1.02 + 0.18 * ((x > 13.2) & (x < 14.35)),
            lambda x: 1.03 + 0.12 * np.sin((x - 16.0) * 5.1) + 0.045 * np.sin((x - 16.0) * 11.5),
            1.0,
        ],
    )


def draw_scene(ax, metrics: dict[str, dict[str, str] | None], style_key: str):
    plt, Arc, Circle, FancyArrowPatch, Polygon, Rectangle = require_matplotlib()
    style = VARIANTS[style_key]
    colors = PALETTES[style.name]
    zmetrics = compact_zone_metrics(metrics)

    ax.set_facecolor(colors["bg"])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 8.7)
    ax.axis("off")

    xs = np.linspace(0, 24, 1000)
    gy = ground_profile(xs)
    ax.fill_between(xs, 0, gy, color=colors["ground"], alpha=0.95, zorder=1)
    ax.plot(xs, gy, color="#475569", lw=style.line_width, zorder=4)

    for x in [4, 8, 12, 16, 20]:
        ax.plot([x, x], [0.72, 7.35], color="#94a3b8", lw=0.75, ls="--", alpha=0.32, zorder=0)

    zone_centers = [2, 6, 10, 14, 18, 22]
    headers = [
        ("Balance", "30 N push recovery", "balance"),
        ("Turn + Roll", "yaw tracking with roll stabilization", "turn"),
        ("Height Tracking", "fixed / step / sine L0", "height"),
        ("Known Terrain", "known height offset -> leg_diff", "known_terrain"),
        ("Terrain Adaptive", "blind feedback on uneven ground", "terrain_adaptive"),
        ("Jump", "smoke tests / exploratory", "jump"),
    ]
    for x, (title, subtitle, key) in zip(zone_centers, headers):
        draw_zone_header(ax, x, title, subtitle, zmetrics[key], colors, style)

    # Balance: push disturbance and recovery.
    draw_robot(ax, 2.0, 1.02, 0.98 * style.robot_scale, 0.72, colors, roll=-0.015)
    ax.add_patch(FancyArrowPatch((0.75, 2.35), (1.45, 2.05), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=colors["red"], zorder=20))
    ax.text(0.68, 2.52, "external push", ha="left", va="center", fontsize=8.8 * style.contrast, color=colors["red"], zorder=21)
    ax.plot([1.15, 2.85], [1.02, 1.02], color=colors["blue"], lw=1.4, alpha=0.75, zorder=5)

    # Turn: path arc and roll cue.
    theta = np.linspace(-0.75, 1.08, 140)
    road_x = 6.0 + 1.45 * np.cos(theta)
    road_y = 2.0 + 0.92 * np.sin(theta)
    ax.plot(road_x, road_y, color=colors["road"], lw=16 * style.contrast, solid_capstyle="round", zorder=2)
    ax.plot(road_x, road_y, color=colors["blue"], lw=1.35 * style.contrast, ls="--", zorder=3)
    draw_robot(ax, 6.05, 1.28, 0.86 * style.robot_scale, 0.70, colors, roll=0.08)
    ax.add_patch(Arc((6.0, 2.34), 1.18, 0.78, theta1=15, theta2=282, color=colors["blue"], lw=1.8 * style.contrast, zorder=20))
    ax.add_patch(FancyArrowPatch((6.48, 2.18), (6.62, 2.5), arrowstyle="-|>", mutation_scale=13, color=colors["blue"], lw=1.4, zorder=20))

    # Height tracking: raise and lower obstacles.
    for px, ph, label in [(8.35, 0.18, "low"), (9.55, 0.46, "raise"), (10.85, 0.24, "lower")]:
        ax.add_patch(Rectangle((px, 1.02), 0.95, ph, facecolor="#d6a85d", edgecolor="#7c4a03", lw=1.0, zorder=5))
        ax.text(px + 0.48, 1.15 + ph, label, ha="center", va="bottom", fontsize=7.7, color="#7c4a03", zorder=8)
    draw_robot(ax, 8.6, 1.22, 0.68 * style.robot_scale, 0.56, colors)
    draw_robot(ax, 9.9, 1.46, 0.72 * style.robot_scale, 0.82, colors)
    draw_robot(ax, 11.15, 1.22, 0.68 * style.robot_scale, 0.50, colors)
    ax.plot([8.65, 11.35], [3.25, 3.25], color=colors["muted"], lw=0.8, alpha=0.45)
    step_x = [8.65, 9.35, 9.35, 10.35, 10.35, 11.35]
    step_y = [3.35, 3.35, 3.78, 3.78, 3.35, 3.35]
    ax.plot(step_x, step_y, color=colors["blue"], lw=1.7 * style.contrast, zorder=10)
    ax.text(10.0, 4.0, "L0 reference", ha="center", va="center", fontsize=8.0 * style.contrast, color=colors["blue"])

    # Known terrain: explicit height offset.
    ax.add_patch(Rectangle((13.0, 1.03), 1.38, 0.42, facecolor="#9ac48f", edgecolor=colors["green"], lw=1.1, zorder=5))
    draw_robot(ax, 14.08, 1.45, 0.84 * style.robot_scale, 0.70, colors, roll=0.025)
    ax.add_patch(FancyArrowPatch((13.12, 2.75), (13.7, 2.1), arrowstyle="<->", mutation_scale=14, lw=1.6, color=colors["green"], zorder=20))
    ax.text(13.52, 2.9, "known Δh", ha="center", va="bottom", fontsize=8.2 * style.contrast, color=colors["green"])

    # Adaptive terrain: unknown roughness and feedback loop.
    rough_x = np.linspace(16.25, 19.75, 160)
    rough_y = ground_profile(rough_x)
    ax.fill_between(rough_x, 1.02, rough_y, color="#94a3b8", alpha=0.42, zorder=5)
    ax.plot(rough_x, rough_y, color=colors["green"], lw=2.1 * style.contrast, zorder=6)
    draw_robot(ax, 18.0, 1.2, 0.84 * style.robot_scale, 0.66, colors, roll=-0.06)
    ax.add_patch(FancyArrowPatch((17.15, 3.03), (17.9, 2.32), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=14, color=colors["orange"], lw=1.6, zorder=20))
    ax.add_patch(FancyArrowPatch((18.1, 2.28), (18.88, 3.03), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=14, color=colors["orange"], lw=1.6, zorder=20))
    ax.text(18.02, 3.23, "adaptive feedback", ha="center", va="bottom", fontsize=8.2 * style.contrast, color=colors["orange"])

    # Jump: explicitly marked as smoke / exploratory.
    ax.add_patch(Rectangle((20.45, 1.0), 0.95, 0.25, facecolor="#b8a16c", edgecolor="#5b4a27", zorder=5))
    ax.add_patch(Rectangle((22.55, 1.0), 0.95, 0.25, facecolor="#b8a16c", edgecolor="#5b4a27", zorder=5))
    ax.fill_between([21.4, 22.55], [1.0, 1.0], [0.36, 0.36], color="#334155", alpha=0.48, zorder=2)
    jump_x = np.linspace(20.9, 23.08, 90)
    jump_y = 1.45 + 0.92 * np.sin((jump_x - jump_x[0]) / (jump_x[-1] - jump_x[0]) * np.pi)
    ax.plot(jump_x, jump_y, color=colors["purple"], lw=1.8 * style.contrast, ls="--", zorder=10)
    draw_robot(ax, 21.1, 1.22, 0.68 * style.robot_scale, 0.58, colors)
    draw_robot(ax, 22.12, 2.24, 0.65 * style.robot_scale, 0.58, colors, roll=0.05)
    draw_robot(ax, 23.0, 1.22, 0.68 * style.robot_scale, 0.58, colors)
    ax.text(22.0, 3.25, "Jump / smoke tests / exploratory", ha="center", va="center", fontsize=8.5 * style.contrast, color=colors["purple"], weight="bold")

    # Bottom development route.
    y_route = 0.42
    ax.plot([1.0, 23.0], [y_route, y_route], color=colors["blue"], lw=2.0 * style.contrast, alpha=0.9, zorder=15)
    ax.add_patch(FancyArrowPatch((22.55, y_route), (23.0, y_route), arrowstyle="-|>", mutation_scale=16, lw=0, color=colors["blue"], zorder=16))
    for x in zone_centers:
        ax.add_patch(Circle((x, y_route), 0.075, facecolor=colors["blue"], edgecolor="white", lw=0.8, zorder=17))
    ax.text(12, 0.12, ROADMAP, ha="center", va="center", fontsize=style.route_size, color=colors["text"], weight="bold")

    # Title hierarchy.
    ax.text(12, 8.45, "Wheel-Legged Robot Capability Map", ha="center", va="center", fontsize=style.title_size, weight="bold", color=colors["text"])
    ax.text(
        12,
        8.1,
        "Physics-guided GP-PMPC: balance, turning, height tracking, terrain and jump smoke",
        ha="center",
        va="center",
        fontsize=10.5 * style.contrast,
        color=colors["muted"],
    )
    ax.text(23.7, 0.1, "code-generated from closeout metrics", ha="right", va="bottom", fontsize=7.8 * style.contrast, color=colors["muted"])


def render_variant(style_key: str, out_dir: Path, metrics: dict[str, dict[str, str] | None]) -> list[Path]:
    plt, *_ = require_matplotlib()
    style = VARIANTS[style_key]
    colors = PALETTES[style.name]
    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)
    fig.patch.set_facecolor(colors["bg"])
    draw_scene(ax, metrics, style_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    png_path = out_dir / style.png_name
    fig.savefig(png_path, dpi=style.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    written.append(png_path)
    if style.pdf_name:
        pdf_path = out_dir / style.pdf_name
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        written.append(pdf_path)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["all", "thesis", "defense", "slide"], default="all")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR / "panorama" / "figures"))
    args = parser.parse_args()

    metrics = load_metrics()
    variants = ["thesis", "defense", "slide"] if args.variant == "all" else [args.variant]
    written: list[Path] = []
    for variant in variants:
        written.extend(render_variant(variant, Path(args.out_dir), metrics))

    for path in written:
        print(f"saved={path.resolve()}")
    for name, row in metrics.items():
        print(f"metric_{name}={'loaded' if row else 'schematic'}")


if __name__ == "__main__":
    main()
