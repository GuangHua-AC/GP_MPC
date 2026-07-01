from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "outputs" / "pmpc_overview" / "metrics" / "pmpc_overview.csv"
FIG_DIR = ROOT / "outputs" / "pmpc_overview" / "figures"
OUT_PNG = FIG_DIR / "pmpc_task_summary.png"
OUT_PDF = FIG_DIR / "pmpc_task_summary.pdf"
TASK_ORDER = ["balance", "turn", "height", "known terrain", "terrain adaptive"]


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_rows() -> list[dict[str, str]]:
    if not IN_CSV.exists():
        raise SystemExit(f"Missing overview CSV. Run: python scripts/pmpc/28_collect_pmpc_overview.py")
    with IN_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    order = {task: i for i, task in enumerate(TASK_ORDER)}
    return sorted(rows, key=lambda row: order.get(row.get("task", ""), 999))


def main() -> None:
    rows = read_rows()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt = require_matplotlib()

    tasks = [row["task"] for row in rows]
    x = np.arange(len(tasks))
    success = np.array([1.0 if str(row.get("success", "")).lower() == "true" else 0.0 for row in rows])
    metric_values = np.array([as_float(row.get("main_tracking_value")) for row in rows])
    theta = np.array([as_float(row.get("max_abs_theta")) for row in rows])
    phi = np.array([as_float(row.get("max_abs_phi")) for row in rows])
    roll = np.array([as_float(row.get("max_abs_roll")) for row in rows])
    plan_time = np.array([as_float(row.get("mean_plan_time_sec")) for row in rows])

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8), constrained_layout=True)
    fig.suptitle("Pure Python GP-PMPC Five-Task Closeout Summary", fontsize=16, weight="bold")

    ax = axes[0, 0]
    colors = ["#16a34a" if ok else "#dc2626" for ok in success]
    ax.bar(x, success, color=colors, edgecolor="#111827", linewidth=0.6)
    ax.set_ylim(0, 1.15)
    ax.set_title("Runs Reached max_steps")
    ax.set_ylabel("success")
    ax.set_xticks(x, tasks, rotation=25, ha="right")
    for i, row in enumerate(rows):
        ax.text(i, success[i] + 0.035, str(row.get("steps", "")), ha="center", fontsize=8)

    ax = axes[0, 1]
    metric_colors = ["#2563eb", "#2563eb", "#2563eb", "#15803d", "#b45309"]
    ax.bar(x, metric_values, color=metric_colors, edgecolor="#111827", linewidth=0.5)
    ax.set_title("Main Tracking / Safety Metric")
    ax.set_xticks(x, tasks, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(rows):
        label = row.get("main_tracking_metric", "")
        value = metric_values[i]
        if not math.isnan(value):
            ax.text(i, value, f"{label}\n{value:.4g}", ha="center", va="bottom", fontsize=7)

    ax = axes[0, 2]
    width = 0.36
    ax.bar(x - width / 2, theta, width, label="max |theta|", color="#60a5fa", edgecolor="#1e3a8a", linewidth=0.4)
    ax.bar(x + width / 2, phi, width, label="max |phi|", color="#f97316", edgecolor="#7c2d12", linewidth=0.4)
    ax.set_title("Pitch / Body Angle Safety")
    ax.set_xticks(x, tasks, rotation=25, ha="right")
    ax.set_ylabel("rad")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(x, roll, color="#22c55e", edgecolor="#14532d", linewidth=0.5)
    ax.set_title("Roll Safety")
    ax.set_xticks(x, tasks, rotation=25, ha="right")
    ax.set_ylabel("rad")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.95, "Most relevant for turn / terrain tasks", transform=ax.transAxes, va="top", fontsize=8, color="#374151")

    ax = axes[1, 1]
    valid = ~np.isnan(plan_time)
    ax.bar(x[valid], plan_time[valid], color="#9333ea", edgecolor="#581c87", linewidth=0.5)
    ax.set_title("Mean Planning Time")
    ax.set_xticks(x, tasks, rotation=25, ha="right")
    ax.set_ylabel("s / control step")
    ax.grid(axis="y", alpha=0.25)
    if not np.any(valid):
        ax.text(0.5, 0.5, "not recorded in some closeout CSVs", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        "Method: Risk-aware GP-PMPC",
        "+ safety-guided action regularization",
        "",
        "Scope: pure Python closeout",
        "Not yet Isaac / residual GP",
        "Jump remains smoke / exploratory",
        "",
        "Capability scene:",
        "outputs/panorama/videos/",
        "capability_scene_final.mp4",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10, color="#111827")

    for out in (OUT_PNG, OUT_PDF):
        fig.savefig(out, dpi=220 if out.suffix == ".png" else 300, bbox_inches="tight", facecolor="white")
        print(f"saved={out}")


if __name__ == "__main__":
    main()
