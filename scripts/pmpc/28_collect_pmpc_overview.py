from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "pmpc_overview" / "metrics"
OUT_CSV = OUT_DIR / "pmpc_overview.csv"

TASKS = [
    {
        "task": "balance",
        "csv": ROOT / "outputs" / "balance" / "metrics" / "balance_pmpc_closeout.csv",
        "metric": "max_abs_theta",
        "prefer": ("GP-PMPC recommended",),
        "notes": "30N/120ms external push; representative risk-aware GP-PMPC closeout.",
    },
    {
        "task": "turn",
        "csv": ROOT / "outputs" / "turn" / "metrics" / "turn_pmpc_closeout.csv",
        "metric": "final_yaw_error_deg",
        "prefer": ("PMPC recommended",),
        "notes": "30deg yaw tracking with roll suppression.",
    },
    {
        "task": "height",
        "csv": ROOT / "outputs" / "height" / "metrics" / "height_pmpc_closeout.csv",
        "metric": "rmse_L0_error_after_1s",
        "prefer": ("PMPC step",),
        "notes": "Representative step L0 tracking; fixed and sine also close out successfully.",
    },
    {
        "task": "known terrain",
        "csv": ROOT / "outputs" / "terrain" / "metrics" / "terrain_pmpc_closeout.csv",
        "metric": "max_abs_leg_diff_error",
        "prefer": ("GP-PMPC recommended seed0",),
        "notes": "Known terrain height difference with leg_diff tracking.",
    },
    {
        "task": "terrain adaptive",
        "csv": ROOT / "outputs" / "terrain_adaptive" / "metrics" / "terrain_adaptive_pmpc_closeout.csv",
        "metric": "max_abs_roll",
        "prefer": ("Adaptive GP-PMPC seed0",),
        "notes": "Blind adaptive leg_diff wrapper without known terrain dH input.",
    },
]

FIELDS = [
    "task",
    "method",
    "final_reason",
    "steps",
    "success",
    "main_tracking_metric",
    "main_tracking_value",
    "max_abs_theta",
    "max_abs_phi",
    "max_abs_roll",
    "max_abs_x",
    "mean_plan_time_sec",
    "total_runtime_sec",
    "recommended",
    "notes",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"missing_closeout_csv={path}")
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def get_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
    return ""


def is_success(row: dict[str, str]) -> bool:
    final_reason = row.get("final_reason", "")
    steps = as_float(row.get("steps"))
    return final_reason == "max_steps" and (math.isnan(steps) or steps >= 1000)


def choose_recommended(rows: list[dict[str, str]], prefer: tuple[str, ...]) -> dict[str, str] | None:
    if not rows:
        return None
    for wanted in prefer:
        for row in rows:
            if row.get("method", "").strip() == wanted:
                return row
    for row in rows:
        haystack = f"{row.get('method', '')} {row.get('notes', '')}".lower()
        if "recommended" in haystack and "pmpc" in haystack and is_success(row):
            return row
    for row in rows:
        if "pmpc" in row.get("method", "").lower() and is_success(row):
            return row
    return rows[0]


def normalize_row(task_cfg: dict[str, object], row: dict[str, str] | None) -> dict[str, object]:
    task = str(task_cfg["task"])
    metric = str(task_cfg["metric"])
    if row is None:
        return {
            "task": task,
            "method": "",
            "final_reason": "missing",
            "steps": "",
            "success": False,
            "main_tracking_metric": metric,
            "main_tracking_value": math.nan,
            "max_abs_theta": math.nan,
            "max_abs_phi": math.nan,
            "max_abs_roll": math.nan,
            "max_abs_x": math.nan,
            "mean_plan_time_sec": math.nan,
            "total_runtime_sec": math.nan,
            "recommended": False,
            "notes": "closeout CSV missing or empty",
        }

    return {
        "task": task,
        "method": row.get("method", ""),
        "final_reason": row.get("final_reason", ""),
        "steps": row.get("steps", ""),
        "success": is_success(row),
        "main_tracking_metric": metric,
        "main_tracking_value": get_value(row, metric),
        "max_abs_theta": get_value(row, "max_abs_theta"),
        "max_abs_phi": get_value(row, "max_abs_phi"),
        "max_abs_roll": get_value(row, "max_abs_roll"),
        "max_abs_x": get_value(row, "max_abs_x"),
        "mean_plan_time_sec": get_value(row, "mean_plan_time_sec"),
        "total_runtime_sec": get_value(row, "total_runtime_sec"),
        "recommended": True,
        "notes": task_cfg["notes"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overview_rows: list[dict[str, object]] = []
    for task_cfg in TASKS:
        rows = read_rows(task_cfg["csv"])  # type: ignore[index]
        selected = choose_recommended(rows, task_cfg["prefer"])  # type: ignore[index]
        overview_rows.append(normalize_row(task_cfg, selected))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(overview_rows)

    print(f"saved={OUT_CSV}")
    for row in overview_rows:
        print(
            f"{row['task']}: method={row['method']} reason={row['final_reason']} "
            f"steps={row['steps']} metric={row['main_tracking_metric']}={row['main_tracking_value']} "
            f"success={row['success']}"
        )


if __name__ == "__main__":
    main()
