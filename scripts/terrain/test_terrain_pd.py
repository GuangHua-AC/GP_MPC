from __future__ import annotations

import argparse
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

from _common import controller_label, save_terrain_result, terrain_filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain-mode", choices=["left_obstacle", "right_obstacle", "sine"], default="left_obstacle")
    parser.add_argument("--v-ref", type=float, default=0.15)
    parser.add_argument("--L0-ref", type=float, default=0.32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--theta0", type=float, default=0.03)
    parser.add_argument("--phi0", type=float, default=0.03)
    parser.add_argument("--x-limit", type=float, default=5.0)
    args = parser.parse_args()

    ensure_dirs()
    env = WheelLeggedEnv(task="terrain", terrain_mode=args.terrain_mode)
    env.p.max_steps = args.steps
    ref = Reference(x_limit=args.x_limit, v_ref=args.v_ref, L0_ref=args.L0_ref)
    controller = WheelLeggedPDController(env)
    state = env.reset(theta0=args.theta0, phi0=args.phi0, L0_init=args.L0_ref, ref=ref)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    left_heights: list[float] = []
    right_heights: list[float] = []
    terrain_diffs: list[float] = []
    support_rolls: list[float] = []
    final_reason = "not_done"

    for _ in range(args.steps):
        action = controller.act(state, ref)
        next_state, reward, done, info = env.step(action, ref)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        left_heights.append(info["left_ground_height"])
        right_heights.append(info["right_ground_height"])
        terrain_diffs.append(info["terrain_diff"])
        support_rolls.append(info["support_roll"])
        state = next_state
        final_reason = info["final_reason"]
        if done:
            break

    out = task_output_subdir("terrain", "pd") / terrain_filename("pd", args.terrain_mode, args.v_ref)
    metrics = save_terrain_result(
        out,
        states=np.asarray(states),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        left_heights=np.asarray(left_heights),
        right_heights=np.asarray(right_heights),
        terrain_diffs=np.asarray(terrain_diffs),
        support_rolls=np.asarray(support_rolls),
        dt=env.p.dt,
        terrain_mode=args.terrain_mode,
        v_ref=args.v_ref,
        controller=controller_label("pd"),
        final_reason=final_reason,
    )
    print("task=terrain_pd")
    print(f"terrain_mode={args.terrain_mode} v_ref={args.v_ref}")
    print(f"steps={metrics['steps']} final_reason={metrics['final_reason']} success={metrics['success']}")
    print(f"max_abs_roll_deg={np.rad2deg(metrics['max_abs_roll_rad']):.3f}")
    print(f"max_abs_support_roll_deg={np.rad2deg(metrics['max_abs_support_roll_rad']):.3f}")
    print(f"max_abs_leg_diff_m={metrics['max_abs_leg_diff_m']:.4f}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
