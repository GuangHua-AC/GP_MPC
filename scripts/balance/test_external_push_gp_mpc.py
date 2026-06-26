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
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--push-force", type=float, default=12.0)
    parser.add_argument("--push-start", type=float, default=1.0)
    parser.add_argument("--push-duration", type=float, default=0.12)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--uncertainty-weight", type=float, default=5.0)
    parser.add_argument("--x-ref", type=float, default=0.0)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--T-limit", type=float, default=None)
    parser.add_argument("--Tp-limit", type=float, default=None)
    args = parser.parse_args()

    ensure_dirs()
    env = WheelLeggedEnv(task="balance")
    if args.T_limit is not None:
        env.p.T_limit = float(args.T_limit)
    if args.Tp_limit is not None:
        env.p.Tp_limit = float(args.Tp_limit)
    ref = Reference(x_ref=args.x_ref, x_limit=args.x_limit, v_ref=0.0)
    model = GPDynamicsModel.load(task_output_subdir("balance", "models") / "balance_gp.joblib")
    cfg = MPCConfig(
        horizon=args.horizon,
        candidates=args.candidates,
        uncertainty_weight=args.uncertainty_weight,
    )
    mpc = RandomShootingMPC(env, model, cfg, WheelLeggedPDController(env))
    state = env.reset(ref=ref)

    states = []
    actions = []
    rewards = []
    push_forces = []
    best_costs = []
    uncertainty_costs = []
    infos = []

    for step in range(args.steps):
        t = step * env.p.dt
        in_push = args.push_start <= t < args.push_start + args.push_duration
        push_force = args.push_force if in_push else 0.0
        env.set_external_force_x(push_force)

        action, plan_info = mpc.plan(state, ref)
        next_state, reward, done, step_info = env.step(action, ref)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        push_forces.append(push_force)
        best_costs.append(plan_info["best_cost"])
        uncertainty_costs.append(plan_info.get("uncertainty_cost", 0.0))
        infos.append({**plan_info, **step_info})
        state = next_state
        if done:
            break

    states_np = np.asarray(states)
    actions_np = np.asarray(actions)
    rewards_np = np.asarray(rewards)
    push_np = np.asarray(push_forces)
    costs_np = np.asarray(best_costs)
    uncertainty_np = np.asarray(uncertainty_costs)

    final_info = infos[-1] if infos else {"final_reason": "no_steps"}
    final_reason = final_info.get("final_reason")
    script_status = "run_completed" if final_reason in {"not_done", "max_steps"} else "terminated_by_env"
    max_abs_theta = float(np.max(np.abs(states_np[:, 0]))) if len(states_np) else 0.0
    max_abs_phi = float(np.max(np.abs(states_np[:, 4]))) if len(states_np) else 0.0
    max_abs_x = float(np.max(np.abs(states_np[:, 2]))) if len(states_np) else 0.0

    force_tag = f"{args.push_force:g}".replace(".", "p").replace("-", "m")
    duration_ms = int(round(args.push_duration * 1000.0))
    t_tag = f"{env.p.T_limit:g}".replace(".", "p").replace("-", "m")
    tp_tag = f"{env.p.Tp_limit:g}".replace(".", "p").replace("-", "m")
    out = task_output_subdir("balance", "mpc") / f"balance_external_push_{force_tag}N_{duration_ms}ms_T{t_tag}_Tp{tp_tag}_gp_mpc.npz"
    np.savez(
        out,
        states=states_np,
        actions=actions_np,
        rewards=rewards_np,
        push_forces=push_np,
        best_costs=costs_np,
        uncertainty_costs=uncertainty_np,
        dt=env.p.dt,
        x_ref=ref.x_ref,
        x_limit=ref.x_limit,
        horizon=args.horizon,
        candidates=args.candidates,
        uncertainty_weight=args.uncertainty_weight,
    )

    print("task=balance_external_push_gp_mpc")
    print(f"steps={len(states)}")
    print(f"horizon={args.horizon} candidates={args.candidates} uncertainty_weight={args.uncertainty_weight}")
    print(f"T_limit={env.p.T_limit} Tp_limit={env.p.Tp_limit}")
    print(f"x_range=[{ref.x_ref - ref.x_limit:.3f}, {ref.x_ref + ref.x_limit:.3f}] m")
    print(f"theta_limit=+/-{env.p.max_abs_theta:.3f} rad phi_limit=+/-{env.p.max_abs_phi:.3f} rad")
    print(f"push_force={args.push_force} N")
    print(f"push_window=[{args.push_start:.3f}, {args.push_start + args.push_duration:.3f}] s")
    print(f"final_reason={final_reason}")
    print(f"script_status={script_status}")
    print(f"max_abs_theta={max_abs_theta:.4f} rad")
    print(f"max_abs_phi={max_abs_phi:.4f} rad")
    print(f"max_abs_x={max_abs_x:.4f} m")
    print(f"final_state={np.round(state, 4)}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
