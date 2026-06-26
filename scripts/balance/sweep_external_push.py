from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from wheel_legged.controllers import WheelLeggedPDController
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv
from wheel_legged.utils.paths import ensure_dirs, task_output_subdir


def run_case(force: float, duration: float, steps: int) -> dict[str, float | str | int]:
    env = WheelLeggedEnv(task="balance")
    ref = Reference(x_ref=0.0, v_ref=0.0)
    controller = WheelLeggedPDController(env)
    state = env.reset(ref=ref)
    states = []
    reason = "not_done"

    for step in range(steps):
        t = step * env.p.dt
        env.set_external_force_x(force if 1.0 <= t < 1.0 + duration else 0.0)
        action = controller.act(state, ref)
        state, _reward, done, info = env.step(action, ref)
        states.append(state.copy())
        reason = info["final_reason"]
        if done:
            break

    states_np = np.asarray(states)
    return {
        "force_N": force,
        "duration_s": duration,
        "steps": len(states),
        "final_reason": reason,
        "max_abs_theta_rad": float(np.max(np.abs(states_np[:, 0]))),
        "max_abs_phi_rad": float(np.max(np.abs(states_np[:, 4]))),
        "max_abs_x_m": float(np.max(np.abs(states_np[:, 2]))),
        "final_theta_rad": float(state[0]),
        "final_phi_rad": float(state[4]),
        "final_x_m": float(state[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forces", nargs="+", type=float, default=[8, 12, 16, 20, 25, 30])
    parser.add_argument("--push-duration", type=float, default=0.12)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()

    ensure_dirs()
    rows = [run_case(force, args.push_duration, args.steps) for force in args.forces]
    out = task_output_subdir("balance", "pd") / "balance_external_push_sweep.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['force_N']:>5g} N | {row['final_reason']:<9} | "
            f"max_theta={row['max_abs_theta_rad']:.4f} rad | "
            f"max_phi={row['max_abs_phi_rad']:.4f} rad | "
            f"max_x={row['max_abs_x_m']:.4f} m"
        )
    print(f"saved={out}")


if __name__ == "__main__":
    main()
