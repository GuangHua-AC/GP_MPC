from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from wheel_legged.utils.paths import OUTPUT_DIR


ROOT = Path(__file__).resolve().parents[2]


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle

    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt, Arc, Circle, FancyArrowPatch, Polygon, Rectangle


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"missing_metrics={path}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str] | None, key: str, default: float = float("nan")) -> float:
    if not row:
        return default
    try:
        value = row.get(key, "")
        return float(value) if value not in {"", "nan", "None"} else default
    except (TypeError, ValueError):
        return default


def first_matching(rows: list[dict[str, str]], *needles: str) -> dict[str, str] | None:
    if not rows:
        return None
    lowered = [needle.lower() for needle in needles]
    for row in rows:
        method = str(row.get("method", "")).lower()
        notes = str(row.get("notes", "")).lower()
        text = f"{method} {notes}"
        if all(needle in text for needle in lowered):
            return row
    return rows[-1]


def load_metrics() -> dict[str, dict[str, str] | None]:
    metrics: dict[str, dict[str, str] | None] = {}
    metrics["balance"] = first_matching(
        read_csv(OUTPUT_DIR / "balance" / "metrics" / "balance_pmpc_closeout.csv"),
        "gp-pmpc",
        "recommended",
    )
    metrics["turn"] = first_matching(
        read_csv(OUTPUT_DIR / "turn" / "metrics" / "turn_pmpc_closeout.csv"),
        "pmpc",
        "recommended",
    )
    metrics["height"] = first_matching(
        read_csv(OUTPUT_DIR / "height" / "metrics" / "height_pmpc_closeout.csv"),
        "pmpc",
        "step",
    )
    metrics["known_terrain"] = first_matching(
        read_csv(OUTPUT_DIR / "terrain" / "metrics" / "terrain_pmpc_closeout.csv"),
        "recommended",
        "seed0",
    )
    metrics["terrain_adaptive"] = first_matching(
        read_csv(OUTPUT_DIR / "terrain_adaptive" / "metrics" / "terrain_adaptive_pmpc_closeout.csv"),
        "adaptive gp-pmpc",
        "seed0",
    )
    jump_rows = read_csv(OUTPUT_DIR / "jump" / "reports" / "jump_2d_xz_pitch_sweep.csv")
    metrics["jump"] = next((row for row in jump_rows if str(row.get("success", "")).lower() == "true"), None)
    if metrics["jump"] is None:
        metrics["jump"] = first_matching(read_csv(OUTPUT_DIR / "jump" / "reports" / "jump_1d_sweep.csv"), "")
    return metrics


def format_metric(value: float, unit: str = "", digits: int = 3) -> str:
    if not np.isfinite(value):
        return "data missing"
    return f"{value:.{digits}f}{unit}"


def zone_metrics(metrics: dict[str, dict[str, str] | None]) -> dict[str, list[str]]:
    b = metrics["balance"]
    t = metrics["turn"]
    h = metrics["height"]
    kt = metrics["known_terrain"]
    ta = metrics["terrain_adaptive"]
    j = metrics["jump"]
    return {
        "balance": [
            f"final: {b.get('final_reason', 'missing') if b else 'data missing'}",
            f"|theta|max {format_metric(as_float(b, 'max_abs_theta'), ' rad')}",
            f"|phi|max {format_metric(as_float(b, 'max_abs_phi'), ' rad')}",
        ],
        "turn": [
            f"yaw err {format_metric(abs(as_float(t, 'final_yaw_error_deg')), ' deg')}",
            f"|roll|max {format_metric(as_float(t, 'max_abs_roll'), ' rad')}",
            f"final: {t.get('final_reason', 'missing') if t else 'data missing'}",
        ],
        "height": [
            f"L0 RMSE {format_metric(as_float(h, 'rmse_L0_error_after_1s'), ' m')}",
            f"|theta|max {format_metric(as_float(h, 'max_abs_theta'), ' rad')}",
            f"mode: {h.get('mode', 'missing') if h else 'data missing'}",
        ],
        "known_terrain": [
            f"leg err {format_metric(as_float(kt, 'max_abs_leg_diff_error'), ' m')}",
            f"|roll|max {format_metric(as_float(kt, 'max_abs_roll'), ' rad')}",
            f"final: {kt.get('final_reason', 'missing') if kt else 'data missing'}",
        ],
        "terrain_adaptive": [
            f"|roll|max {format_metric(as_float(ta, 'max_abs_roll'), ' rad')}",
            f"|leg_diff|max {format_metric(as_float(ta, 'max_abs_leg_diff'), ' m')}",
            "blind feedback",
        ],
        "jump": [
            "1D / 2D / pitch smoke",
            f"h target {format_metric(as_float(j, 'h_target'), ' m')}",
            f"success: {j.get('success', 'data missing') if j else 'data missing'}",
        ],
    }


def style_palette(style: str) -> dict[str, str]:
    if style == "dark":
        return {
            "bg": "#111827",
            "panel": "#1f2937",
            "text": "#f9fafb",
            "muted": "#cbd5e1",
            "ground": "#475569",
            "road": "#334155",
            "accent": "#38bdf8",
            "orange": "#fb923c",
            "green": "#22c55e",
            "red": "#f87171",
            "robot": "#e5e7eb",
        }
    if style == "clean":
        return {
            "bg": "#ffffff",
            "panel": "#f8fafc",
            "text": "#111827",
            "muted": "#64748b",
            "ground": "#d6d3d1",
            "road": "#e5e7eb",
            "accent": "#2563eb",
            "orange": "#d97706",
            "green": "#16a34a",
            "red": "#dc2626",
            "robot": "#f3f4f6",
        }
    return {
        "bg": "#f7f4ee",
        "panel": "#fffaf0",
        "text": "#1f2937",
        "muted": "#6b7280",
        "ground": "#c7b299",
        "road": "#d9d2c3",
        "accent": "#1d4ed8",
        "orange": "#b45309",
        "green": "#047857",
        "red": "#b91c1c",
        "robot": "#f8fafc",
    }


def draw_robot(ax, x: float, y: float, scale: float = 1.0, body_height: float = 0.62, roll: float = 0.0, label: str | None = None, colors=None):
    from matplotlib.patches import Circle, Rectangle

    colors = colors or {}
    body_w = 0.75 * scale
    body_h = 0.32 * scale
    wheel_r = 0.14 * scale
    wheel_dx = 0.28 * scale
    wheel_y = y + wheel_r
    body_y = y + body_height * scale
    tilt = math.sin(roll) * 0.08 * scale
    body = Rectangle((x - body_w / 2, body_y - body_h / 2), body_w, body_h, angle=np.rad2deg(roll), facecolor=colors.get("robot", "#f8fafc"), edgecolor="#111827", lw=1.3, zorder=7)
    ax.add_patch(body)
    for side in [-1, 1]:
        wx = x + side * wheel_dx
        ax.add_patch(Circle((wx, wheel_y), wheel_r, facecolor="#ffffff", edgecolor="#111827", lw=1.4, zorder=7))
        ax.plot([wx, x + side * wheel_dx * 0.8], [wheel_y, body_y - body_h * 0.45 + side * tilt], color="#111827", lw=1.7, zorder=7)
    ax.plot([x - wheel_dx, x + wheel_dx], [wheel_y - wheel_r, wheel_y - wheel_r], color="#111827", lw=1.0, alpha=0.5, zorder=6)
    if label:
        ax.text(x, body_y + 0.35 * scale, label, ha="center", va="bottom", fontsize=8, color=colors.get("text", "#111827"), zorder=9)


def draw_zone_label(ax, x: float, title_cn: str, subtitle: str, metrics: list[str], colors, show_metrics: bool):
    ax.text(x, 8.18, title_cn, ha="center", va="center", fontsize=16, weight="bold", color=colors["text"])
    ax.text(x, 7.82, subtitle, ha="center", va="center", fontsize=9.5, color=colors["muted"])
    if show_metrics:
        ax.text(
            x,
            7.36,
            "\n".join(metrics),
            ha="center",
            va="top",
            fontsize=8.2,
            color=colors["text"],
            bbox=dict(boxstyle="round,pad=0.32", facecolor=colors["panel"], edgecolor="none", alpha=0.88),
        )


def draw_curved_arrow(ax, start, end, color, label=None, rad=0.2):
    _, _, FancyArrowPatch, *_ = require_matplotlib()
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=12, lw=1.8, color=color, zorder=8)
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + abs(rad)
        ax.text(mx, my, label, fontsize=8, color=color, ha="center")


def draw_metric_curve(ax, x: float, y: float, w: float, h: float, colors, kind: str = "sine"):
    xs = np.linspace(0, 1, 80)
    if kind == "step":
        ys = np.where(xs < 0.45, 0.25, 0.78)
    else:
        ys = 0.5 + 0.32 * np.sin(2 * np.pi * xs)
    ax.plot(x + xs * w, y + ys * h, color=colors["accent"], lw=1.7, zorder=9)
    ax.plot([x, x + w], [y, y], color=colors["muted"], lw=0.6, alpha=0.6)
    ax.plot([x, x], [y, y + h], color=colors["muted"], lw=0.6, alpha=0.6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUTPUT_DIR / "panorama" / "figures" / "wheel_legged_task_big_map_v2.png"))
    parser.add_argument("--pdf", default=str(OUTPUT_DIR / "panorama" / "figures" / "wheel_legged_task_big_map_v2.pdf"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--style", choices=["clean", "dark", "thesis"], default="thesis")
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    plt, Arc, Circle, FancyArrowPatch, Polygon, Rectangle = require_matplotlib()
    colors = style_palette(args.style)
    metrics = load_metrics()
    zmetrics = zone_metrics(metrics)

    fig, ax = plt.subplots(figsize=(21, 9), dpi=args.dpi)
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Continuous ground band.
    xs = np.linspace(0, 24, 800)
    ground = np.piecewise(
        xs,
        [xs < 4, (xs >= 4) & (xs < 8), (xs >= 8) & (xs < 12), (xs >= 12) & (xs < 16), (xs >= 16) & (xs < 20), xs >= 20],
        [
            1.05,
            lambda x: 1.05 + 0.18 * np.sin((x - 4) / 4 * np.pi),
            lambda x: 1.0 + 0.18 * (x > 9.5) - 0.13 * (x > 10.9),
            lambda x: 1.02 + 0.22 * ((x > 13.15) & (x < 14.2)),
            lambda x: 1.02 + 0.12 * np.sin((x - 16) * 4.8) + 0.05 * np.sin((x - 16) * 11.0),
            lambda x: 1.0,
        ],
    )
    ax.fill_between(xs, 0, ground, color=colors["ground"], alpha=0.92, zorder=1)
    ax.plot(xs, ground, color="#4b5563", lw=1.2, zorder=2)

    zone_edges = [0, 4, 8, 12, 16, 20, 24]
    zone_centers = [2, 6, 10, 14, 18, 22]
    for edge in zone_edges[1:-1]:
        ax.plot([edge, edge], [0.35, 8.55], color=colors["muted"], lw=0.8, alpha=0.22, ls="--")

    # Zone 1: balance and push.
    draw_robot(ax, 2.05, 1.05, scale=1.05, body_height=0.82, colors=colors)
    ax.add_patch(FancyArrowPatch((0.8, 2.35), (1.55, 2.1), arrowstyle="-|>", mutation_scale=22, color=colors["red"], lw=2.4))
    ax.text(0.7, 2.55, "30N push", color=colors["red"], fontsize=9, ha="left")

    # Zone 2: turn.
    theta = np.linspace(-0.8, 1.1, 140)
    cx, cy = 6.0, 1.0
    road_x = cx + 1.5 * np.cos(theta)
    road_y = cy + 1.05 * np.sin(theta) + 1.0
    ax.plot(road_x, road_y, color=colors["road"], lw=18, solid_capstyle="round", zorder=2)
    ax.plot(road_x, road_y, color=colors["accent"], lw=1.4, ls="--", zorder=3)
    draw_robot(ax, 6.0, 1.32, scale=0.92, body_height=0.78, roll=0.08, colors=colors)
    ax.add_patch(Arc((6.0, 2.25), 1.2, 0.8, theta1=10, theta2=285, color=colors["accent"], lw=2.0))
    ax.add_patch(FancyArrowPatch((6.48, 2.08), (6.6, 2.42), arrowstyle="-|>", mutation_scale=13, color=colors["accent"], lw=1.4))

    # Zone 3: height.
    for px, py, pw, ph in [(8.2, 1.05, 1.2, 0.18), (9.55, 1.05, 1.2, 0.46), (10.9, 1.05, 1.2, 0.25)]:
        ax.add_patch(Rectangle((px, py), pw, ph, facecolor="#d6a85d", edgecolor="#7c4a03", lw=1.0, zorder=3))
    draw_robot(ax, 8.8, 1.24, scale=0.78, body_height=0.62, colors=colors)
    draw_robot(ax, 10.15, 1.52, scale=0.78, body_height=0.82, colors=colors)
    draw_robot(ax, 11.45, 1.3, scale=0.78, body_height=0.55, colors=colors)
    draw_metric_curve(ax, 9.0, 3.3, 2.25, 0.75, colors, "step")
    ax.text(10.12, 4.15, "L0_ref", fontsize=8, color=colors["accent"], ha="center")

    # Zone 4: known terrain.
    ax.add_patch(Rectangle((13.0, 1.06), 1.35, 0.45, facecolor="#8bbf88", edgecolor="#2f6b2f", lw=1.0, zorder=3))
    ax.text(13.68, 1.62, "known Δh", color=colors["green"], fontsize=8, ha="center")
    draw_robot(ax, 14.1, 1.52, scale=0.9, body_height=0.78, roll=0.02, colors=colors)
    ax.add_patch(FancyArrowPatch((13.1, 2.8), (13.7, 2.15), arrowstyle="<->", mutation_scale=13, color=colors["green"], lw=1.6))
    ax.text(13.25, 2.9, "leg_diff", fontsize=8, color=colors["green"])

    # Zone 5: adaptive terrain.
    rough_x = np.linspace(16.2, 19.7, 160)
    rough_y = 1.05 + 0.16 * np.sin((rough_x - 16.2) * 5.0) + 0.06 * np.sin((rough_x - 16.2) * 12.0)
    ax.fill_between(rough_x, 1.02, rough_y, color="#9ca3af", alpha=0.55, zorder=3)
    ax.plot(rough_x, rough_y, color=colors["green"], lw=2.0, zorder=4)
    draw_robot(ax, 18.0, 1.2, scale=0.9, body_height=0.72, roll=-0.06, colors=colors)
    ax.add_patch(FancyArrowPatch((17.1, 3.0), (18.0, 2.25), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=14, color=colors["orange"], lw=1.7))
    ax.add_patch(FancyArrowPatch((18.1, 2.25), (18.85, 3.0), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=14, color=colors["orange"], lw=1.7))
    ax.text(18.0, 3.25, "feedback loop", color=colors["orange"], fontsize=8, ha="center")

    # Zone 6: jump smoke.
    ax.add_patch(Rectangle((20.4, 1.0), 1.0, 0.25, facecolor="#b7a47a", edgecolor="#5b4a27", zorder=3))
    ax.add_patch(Rectangle((22.5, 1.0), 1.0, 0.25, facecolor="#b7a47a", edgecolor="#5b4a27", zorder=3))
    ax.fill_between([21.4, 22.5], [1.0, 1.0], [0.35, 0.35], color="#334155", alpha=0.45, zorder=2)
    jump_x = np.linspace(20.85, 23.0, 80)
    jump_y = 1.6 + 1.0 * np.sin((jump_x - 20.85) / (23.0 - 20.85) * np.pi)
    ax.plot(jump_x, jump_y, color=colors["accent"], lw=1.8, ls="--", zorder=5)
    draw_robot(ax, 21.1, 1.25, scale=0.75, body_height=0.65, colors=colors)
    draw_robot(ax, 22.1, 2.3, scale=0.72, body_height=0.65, roll=0.04, colors=colors)
    draw_robot(ax, 23.0, 1.25, scale=0.75, body_height=0.65, colors=colors)

    titles = [
        ("平衡控制", "Balance / 30N 外力推扰", "balance"),
        ("转向控制", "Turn / yaw_ref = 30 deg", "turn"),
        ("变腿高", "Height / fixed-step-sine", "height"),
        ("已知地形", "Known Terrain / leg_diff tracking", "known_terrain"),
        ("地形自适应", "Terrain Adaptive / blind feedback", "terrain_adaptive"),
        ("跳跃探索", "Jump / smoke tests", "jump"),
    ]
    for x, (cn, subtitle, key) in zip(zone_centers, titles):
        draw_zone_label(ax, x, cn, subtitle, zmetrics[key], colors, args.show_metrics)

    # Bottom roadmap.
    roadmap = "平衡 -> 转向 -> 变腿高 -> 已知地形 -> 地形自适应 -> 跳跃"
    ax.plot([1.1, 22.9], [0.42, 0.42], color=colors["accent"], lw=2.0, alpha=0.8)
    for x in zone_centers:
        ax.add_patch(Circle((x, 0.42), 0.08, facecolor=colors["accent"], edgecolor="none", zorder=6))
    ax.text(12, 0.12, roadmap, ha="center", va="center", fontsize=12, color=colors["text"])

    ax.text(12, 8.77, "双轮足机器人任务大地图", ha="center", va="center", fontsize=22, weight="bold", color=colors["text"])
    ax.text(12, 8.48, "wheel_legged_new capability map", ha="center", va="center", fontsize=12, color=colors["muted"])
    ax.text(23.75, 0.15, "code-generated, reproducible", ha="right", va="bottom", fontsize=8, color=colors["muted"])

    out = Path(args.out)
    pdf = Path(args.pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved_png={out.resolve()}")
    print(f"saved_pdf={pdf.resolve()}")
    for name, row in metrics.items():
        print(f"metric_{name}={'loaded' if row else 'schematic'}")


if __name__ == "__main__":
    main()
