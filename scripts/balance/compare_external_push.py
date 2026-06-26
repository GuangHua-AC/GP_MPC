from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from wheel_legged.utils.paths import ensure_dirs, task_output_dir


PATTERN = re.compile(
    r"balance_external_push_(?P<force>[\dm\.p]+)N_(?P<duration>\d+)ms"
    r"(?:_T(?P<T>[\dm\.p]+)_Tp(?P<Tp>[\dm\.p]+))?"
    r"_(?P<method>pd|nn_mpc_torch|nn_mpc_sklearn|gp_mpc)\.npz$"
)


def parse_num(text: str | None, default: float) -> float:
    if not text:
        return default
    return float(text.replace("p", ".").replace("m", "-"))


def infer_method(path: Path, method_tag: str) -> str:
    if method_tag == "pd":
        return "PD"
    if method_tag.startswith("nn_mpc"):
        return "NN-MPC"
    if method_tag == "gp_mpc":
        return "GP-MPC"
    return path.stem


def infer_success(states: np.ndarray, x_limit: float = 2.0, x_ref: float = 0.0) -> tuple[str, bool]:
    if len(states) == 0:
        return "no_steps", False
    theta = states[-1, 0]
    x = states[-1, 2]
    phi = states[-1, 4]
    max_theta = float(np.max(np.abs(states[:, 0])))
    max_phi = float(np.max(np.abs(states[:, 4])))
    max_x = float(np.max(np.abs(states[:, 2])))
    if len(states) < 1199 and max_phi > 0.75:
        return "fall_phi", False
    if len(states) < 1199 and max_theta > 0.75:
        return "fall_theta", False
    if max_theta > 0.80:
        return "fall_theta", False
    if max_phi > 0.80:
        return "fall_phi", False
    if max_x > max(5.0, abs(x_ref) + x_limit + 1.0):
        return "x_out", False
    recovered = abs(theta) < 0.05 and abs(phi) < 0.05 and abs(x - x_ref) <= x_limit
    return "max_steps" if recovered else "not_fully_recovered", bool(recovered)


def summarize_file(path: Path) -> dict[str, str | float | int | bool] | None:
    match = PATTERN.search(path.name)
    if not match:
        return None
    if match.group("method") == "pd" and (match.group("T") is None or match.group("Tp") is None):
        return None

    data = np.load(path)
    if "states" not in data:
        return None

    states = data["states"]
    actions = data["actions"] if "actions" in data else np.zeros((len(states), 6))
    method = infer_method(path, match.group("method"))
    force = parse_num(match.group("force"), 0.0)
    duration_s = int(match.group("duration")) / 1000.0
    T_limit = parse_num(match.group("T"), 1.2)
    Tp_limit = parse_num(match.group("Tp"), 1.2)
    x_limit = float(data["x_limit"]) if "x_limit" in data.files else 2.0
    x_ref = float(data["x_ref"]) if "x_ref" in data.files else 0.0
    reason, success = infer_success(states, x_limit=x_limit, x_ref=x_ref)

    max_abs_theta = float(np.max(np.abs(states[:, 0]))) if len(states) else 0.0
    max_abs_phi = float(np.max(np.abs(states[:, 4]))) if len(states) else 0.0
    max_abs_x = float(np.max(np.abs(states[:, 2]))) if len(states) else 0.0
    final = states[-1] if len(states) else np.zeros(14)
    max_abs_action = np.max(np.abs(actions), axis=0) if len(actions) else np.zeros(6)

    return {
        "method": method,
        "push_force_N": force,
        "push_duration_s": duration_s,
        "T_limit": T_limit,
        "Tp_limit": Tp_limit,
        "x_ref": x_ref,
        "x_limit": x_limit,
        "steps": int(len(states)),
        "inferred_final_reason": reason,
        "success_recovered": success,
        "max_abs_theta_rad": max_abs_theta,
        "max_abs_phi_rad": max_abs_phi,
        "max_abs_x_m": max_abs_x,
        "final_theta_rad": float(final[0]),
        "final_phi_rad": float(final[4]),
        "final_x_m": float(final[2]),
        "max_abs_T": float(max_abs_action[0]),
        "max_abs_Tp": float(max_abs_action[1]),
        "file": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_dirs()
    balance_dir = task_output_dir("balance")
    search_roots = [balance_dir / "pd", balance_dir / "mpc", balance_dir / "final"]
    paths = []
    for root in search_roots:
        if root.exists():
            paths += sorted(root.rglob("balance_external_push*.npz"))
    deduped: dict[str, Path] = {}
    final_root = balance_dir / "final"
    for path in paths:
        old = deduped.get(path.name)
        if old is None:
            deduped[path.name] = path
            continue
        if final_root in path.parents and final_root not in old.parents:
            deduped[path.name] = path
    paths = sorted(deduped.values())

    rows = []
    for path in paths:
        row = summarize_file(path)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: (float(r["push_force_N"]), float(r["Tp_limit"]), str(r["method"])))

    out = Path(args.out) if args.out else balance_dir / "metrics" / "balance_external_push_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No balance external push result files found.")
        return

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={out}")
    print("method, force, Tp_limit, success, reason, max_theta, max_phi, max_x")
    for row in rows:
        print(
            f"{row['method']}, {row['push_force_N']:g}N, Tp={row['Tp_limit']:g}, "
            f"success={row['success_recovered']}, reason={row['inferred_final_reason']}, "
            f"theta={row['max_abs_theta_rad']:.4f}, "
            f"phi={row['max_abs_phi_rad']:.4f}, "
            f"x={row['max_abs_x_m']:.4f}"
        )


if __name__ == "__main__":
    main()
