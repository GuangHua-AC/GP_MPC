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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="balance_turn_roll", choices=["balance", "balance_turn", "balance_turn_roll", "height", "terrain"])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--yaw-deg", type=float, default=20.0)
    parser.add_argument("--v-ref", type=float, default=None)
    parser.add_argument("--x-limit", type=float, default=None)
    parser.add_argument("--L0-ref", type=float, default=0.34)
    parser.add_argument("--theta0", type=float, default=0.03)
    parser.add_argument("--phi0", type=float, default=0.03)
    parser.add_argument("--x0", type=float, default=0.0)
    args = parser.parse_args()

    ensure_dirs()
    env = WheelLeggedEnv(task=args.task)
    env.p.max_steps = args.steps
    if args.v_ref is None:
        v_ref = 0.15 if args.task in {"balance_turn", "balance_turn_roll", "terrain"} else 0.0
    else:
        v_ref = args.v_ref
    if args.x_limit is None:
        x_limit = 2.0 if args.task in {"balance", "height"} else 5.0
    else:
        x_limit = args.x_limit

    ref = Reference(yaw_ref=np.deg2rad(args.yaw_deg), x_limit=x_limit, v_ref=v_ref, L0_ref=args.L0_ref)
    controller = WheelLeggedPDController(env)
    state = env.reset(theta0=args.theta0, phi0=args.phi0, x0=args.x0, ref=ref)
    states = []
    actions = []
    rewards = []
    info = {}

    for _ in range(args.steps):
        action = controller.act(state, ref)
        next_state, reward, done, info = env.step(action, ref)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        if done:
            break

    out = task_output_subdir(args.task, "pd") / f"{args.task}_pd.npz"
    np.savez(
        out,
        states=np.asarray(states),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        dt=env.p.dt,
        x_ref=ref.x_ref,
        x_limit=ref.x_limit,
        v_ref=ref.v_ref,
        yaw_ref=ref.yaw_ref,
        L0_ref=ref.L0_ref,
    )
    print(f"task={args.task}")
    print(f"steps={len(states)}")
    print(f"final_reason={info.get('final_reason')}")
    print(f"final_state={np.round(state, 4)}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
