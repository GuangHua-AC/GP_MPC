from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from wheel_legged.controllers import RandomShootingMPC, WheelLeggedPDController
from wheel_legged.controllers.mpc import MPCConfig
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv
from wheel_legged.models import GPDynamicsModel
from wheel_legged.utils.paths import ensure_dirs, task_output_subdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="terrain")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--uncertainty-weight", type=float, default=5.0)
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
    model = GPDynamicsModel.load(task_output_subdir(args.task, "models") / f"{args.task}_gp.joblib")
    cfg = MPCConfig(horizon=args.horizon, candidates=args.candidates, uncertainty_weight=args.uncertainty_weight)
    mpc = RandomShootingMPC(env, model, cfg, WheelLeggedPDController(env))
    if args.v_ref is None:
        v_ref = 0.15 if args.task in {"balance_turn", "balance_turn_roll", "terrain"} else 0.0
    else:
        v_ref = args.v_ref
    if args.x_limit is None:
        x_limit = 2.0 if args.task in {"balance", "height"} else 5.0
    else:
        x_limit = args.x_limit
    ref = Reference(yaw_ref=np.deg2rad(args.yaw_deg), x_limit=x_limit, v_ref=v_ref, L0_ref=args.L0_ref)
    state = env.reset(theta0=args.theta0, phi0=args.phi0, x0=args.x0, ref=ref)
    states = []
    actions = []
    infos = []

    for _ in range(args.steps):
        action, info = mpc.plan(state, ref)
        state, reward, done, step_info = env.step(action, ref)
        states.append(state)
        actions.append(action)
        infos.append({**info, **step_info, "reward": reward})
        if done:
            break

    out = task_output_subdir(args.task, "mpc") / f"{args.task}_gp_mpc.npz"
    np.savez(
        out,
        states=np.asarray(states),
        actions=np.asarray(actions),
        dt=env.p.dt,
        x_ref=ref.x_ref,
        x_limit=ref.x_limit,
        v_ref=ref.v_ref,
        yaw_ref=ref.yaw_ref,
        L0_ref=ref.L0_ref,
    )
    print(f"saved={out}")
    print(f"steps={len(states)} final_reason={infos[-1].get('final_reason') if infos else 'none'}")


if __name__ == "__main__":
    main()
