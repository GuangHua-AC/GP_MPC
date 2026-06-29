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
from wheel_legged.models import GPDynamicsModel
from wheel_legged.pmpc import RiskAwareMPCConfig, RiskAwareShootingMPC
from wheel_legged.utils.paths import ensure_dirs, task_output_subdir


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def default_output_path(args, env: WheelLeggedEnv) -> Path:
    force_tag = _tag(args.push_force)
    duration_ms = int(round(args.push_duration * 1000.0))
    t_tag = _tag(env.p.T_limit)
    tp_tag = _tag(env.p.Tp_limit)
    uw_tag = _tag(args.uncertainty_weight)
    cw_tag = _tag(args.chance_weight)
    gw_tag = _tag(args.guide_weight)
    ks_tag = _tag(args.k_sigma)
    tw_tag = _tag(args.terminal_weight)
    seed_tag = _tag(args.seed)
    ps_tag = _tag(args.push_start)
    return (
        task_output_subdir("balance", "pmpc")
        / (
            f"balance_external_push_{force_tag}N_{duration_ms}ms_T{t_tag}_Tp{tp_tag}_"
            f"Uw{uw_tag}_Cw{cw_tag}_Gw{gw_tag}_K{ks_tag}_Tw{tw_tag}_"
            f"seed{seed_tag}_pushStart{ps_tag}_gp_pmpc.npz"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--uncertainty-weight", type=float, default=5.0)
    parser.add_argument("--chance-weight", type=float, default=50.0)
    parser.add_argument("--guide-weight", type=float, default=0.0)
    parser.add_argument("--k-sigma", type=float, default=2.0)
    parser.add_argument("--terminal-weight", type=float, default=0.0)
    parser.add_argument("--push-force", type=float, default=30.0)
    parser.add_argument("--push-duration", type=float, default=0.12)
    parser.add_argument("--push-start", type=float, default=1.0)
    parser.add_argument("--T-limit", type=float, default=1.2)
    parser.add_argument("--Tp-limit", type=float, default=1.5)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    model_path = task_output_subdir("balance", "models") / "balance_gp.joblib"
    if not model_path.exists():
        print(f"missing_model={model_path}")
        print("please run: python scripts/common/05_train_gp.py --task balance --max-points 1500")
        raise SystemExit(2)

    env = WheelLeggedEnv(task="balance")
    env.p.T_limit = float(args.T_limit)
    env.p.Tp_limit = float(args.Tp_limit)
    ref = Reference(x_ref=0.0, x_limit=args.x_limit, v_ref=0.0)
    model = GPDynamicsModel.load(model_path)
    cfg = RiskAwareMPCConfig(
        horizon=args.horizon,
        candidates=args.candidates,
        seed=args.seed,
        uncertainty_weight=args.uncertainty_weight,
        chance_weight=args.chance_weight,
        guide_weight=args.guide_weight,
        k_sigma=args.k_sigma,
        terminal_weight=args.terminal_weight,
    )
    mpc = RiskAwareShootingMPC(env, model, cfg, WheelLeggedPDController(env))
    state = env.reset(ref=ref)

    states = []
    actions = []
    rewards = []
    push_forces = []
    best_costs = []
    mean_costs = []
    uncertainty_costs = []
    chance_penalties = []
    terminal_costs = []
    guide_costs = []
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
        mean_costs.append(plan_info["mean_cost"])
        uncertainty_costs.append(plan_info["uncertainty_cost"])
        chance_penalties.append(plan_info["chance_penalty"])
        terminal_costs.append(plan_info["terminal_cost"])
        guide_costs.append(plan_info["guide_cost"])
        infos.append({**plan_info, **step_info})
        if step % 100 == 0:
            guide_norm = float(np.linalg.norm(plan_info["guide_action"]))
            best_norm = float(np.linalg.norm(plan_info["best_first_action"]))
            print(
                f"debug step={step} theta={state[0]:+.4f} phi={state[4]:+.4f} x={state[2]:+.4f} "
                f"best_cost={plan_info['best_cost']:.3f} guide_action_norm={guide_norm:.3f} "
                f"best_action_norm={best_norm:.3f}"
            )
        state = next_state
        if done:
            break

    states_np = np.asarray(states, dtype=float)
    actions_np = np.asarray(actions, dtype=float)
    rewards_np = np.asarray(rewards, dtype=float)
    push_np = np.asarray(push_forces, dtype=float)
    final_info = infos[-1] if infos else {"final_reason": "no_steps"}
    final_reason = str(final_info.get("final_reason", "no_steps"))
    max_abs_theta = float(np.max(np.abs(states_np[:, 0]))) if len(states_np) else 0.0
    max_abs_phi = float(np.max(np.abs(states_np[:, 4]))) if len(states_np) else 0.0
    max_abs_x = float(np.max(np.abs(states_np[:, 2]))) if len(states_np) else 0.0

    out = Path(args.out) if args.out else default_output_path(args, env)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        states=states_np,
        actions=actions_np,
        rewards=rewards_np,
        push_forces=push_np,
        best_costs=np.asarray(best_costs, dtype=float),
        mean_costs=np.asarray(mean_costs, dtype=float),
        uncertainty_costs=np.asarray(uncertainty_costs, dtype=float),
        chance_penalties=np.asarray(chance_penalties, dtype=float),
        terminal_costs=np.asarray(terminal_costs, dtype=float),
        guide_costs=np.asarray(guide_costs, dtype=float),
        guide_action_norms=np.asarray([np.linalg.norm(info["guide_action"]) for info in infos], dtype=float),
        best_action_norms=np.asarray([np.linalg.norm(info["best_first_action"]) for info in infos], dtype=float),
        dt=env.p.dt,
        x_ref=ref.x_ref,
        x_limit=ref.x_limit,
        v_ref=ref.v_ref,
        yaw_ref=ref.yaw_ref,
        L0_ref=ref.L0_ref,
        horizon=args.horizon,
        candidates=args.candidates,
        push_force=args.push_force,
        push_duration=args.push_duration,
        push_start=args.push_start,
        uncertainty_weight=args.uncertainty_weight,
        chance_weight=args.chance_weight,
        guide_weight=args.guide_weight,
        k_sigma=args.k_sigma,
        terminal_weight=args.terminal_weight,
        T_limit=env.p.T_limit,
        Tp_limit=env.p.Tp_limit,
        final_reason=np.array([final_reason]),
    )

    print("task=balance_external_push_gp_pmpc")
    print(f"saved={out}")
    print(f"steps={len(states_np)}")
    print(f"final_reason={final_reason}")
    print(f"max_abs_theta={max_abs_theta:.4f}")
    print(f"max_abs_phi={max_abs_phi:.4f}")
    print(f"max_abs_x={max_abs_x:.4f}")


if __name__ == "__main__":
    main()
