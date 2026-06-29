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
from wheel_legged.utils.paths import OUTPUT_DIR, ensure_dirs, task_output_subdir


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _angle_error(angle: np.ndarray | float, ref: np.ndarray | float) -> np.ndarray | float:
    return (angle - ref + np.pi) % (2.0 * np.pi) - np.pi


def default_output_path(args, env: WheelLeggedEnv) -> Path:
    target_tag = _tag(args.target_deg)
    v_tag = _tag(args.v_ref)
    uw_tag = _tag(args.uncertainty_weight)
    cw_tag = _tag(args.chance_weight)
    gw_tag = _tag(args.guide_weight)
    ks_tag = _tag(args.k_sigma)
    tw_tag = _tag(args.terminal_weight)
    seed_tag = _tag(args.seed)
    t_tag = _tag(env.p.T_limit)
    tp_tag = _tag(env.p.Tp_limit)
    return (
        OUTPUT_DIR
        / "turn"
        / "pmpc"
        / (
            f"turn_pmpc_target{target_tag}deg_v{v_tag}_T{t_tag}_Tp{tp_tag}_"
            f"Uw{uw_tag}_Cw{cw_tag}_Gw{gw_tag}_K{ks_tag}_Tw{tw_tag}_seed{seed_tag}.npz"
        )
    )


def _missing_model_message(model_path: Path) -> str:
    return (
        f"Missing balance_turn_roll GP model: {model_path}\n"
        "Please run:\n"
        "python scripts/common/02_collect_data.py --task balance_turn_roll --episodes 120 --steps 600 --noise-scale 0.06\n"
        "python scripts/common/05_train_gp.py --task balance_turn_roll --max-points 1500"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--uncertainty-weight", type=float, default=5.0)
    parser.add_argument("--chance-weight", type=float, default=20.0)
    parser.add_argument("--guide-weight", type=float, default=20.0)
    parser.add_argument("--terminal-weight", type=float, default=0.0)
    parser.add_argument("--k-sigma", type=float, default=2.0)
    parser.add_argument("--target-deg", type=float, default=30.0)
    parser.add_argument("--v-ref", type=float, default=0.15)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--T-limit", type=float, default=1.2)
    parser.add_argument("--Tp-limit", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_dirs()
    model_path = task_output_subdir("balance_turn_roll", "models") / "balance_turn_roll_gp.joblib"
    if not model_path.exists():
        print(_missing_model_message(model_path))
        raise SystemExit(2)

    env = WheelLeggedEnv(task="balance_turn_roll")
    env.p.max_steps = args.steps
    env.p.T_limit = float(args.T_limit)
    env.p.Tp_limit = float(args.Tp_limit)
    ref = Reference(yaw_ref=np.deg2rad(args.target_deg), v_ref=args.v_ref, x_limit=args.x_limit)

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
        chance_enabled=("theta", "phi", "roll", "x"),
    )
    mpc = RiskAwareShootingMPC(env, model, cfg, WheelLeggedPDController(env))
    state = env.reset(ref=ref)

    states = []
    actions = []
    rewards = []
    yaw_refs = []
    roll_refs = []
    best_costs = []
    mean_costs = []
    uncertainty_costs = []
    chance_penalties = []
    guide_costs = []
    terminal_costs = []
    infos = []

    final_reason = "not_done"
    for step in range(args.steps):
        action, plan_info = mpc.plan(state, ref)
        next_state, reward, done, step_info = env.step(action, ref)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        yaw_refs.append(ref.yaw_ref)
        roll_refs.append(ref.roll_ref)
        best_costs.append(plan_info["best_cost"])
        mean_costs.append(plan_info["mean_cost"])
        uncertainty_costs.append(plan_info["uncertainty_cost"])
        chance_penalties.append(plan_info["chance_penalty"])
        guide_costs.append(plan_info["guide_cost"])
        terminal_costs.append(plan_info["terminal_cost"])
        infos.append({**plan_info, **step_info})

        if step % 100 == 0:
            yaw_err_deg = float(np.rad2deg(_angle_error(state[6], ref.yaw_ref)))
            print(
                f"debug step={step} yaw={np.rad2deg(state[6]):+.2f}deg "
                f"yaw_err={yaw_err_deg:+.2f}deg roll={np.rad2deg(state[8]):+.2f}deg "
                f"theta={state[0]:+.4f} phi={state[4]:+.4f} x={state[2]:+.4f} "
                f"best_cost={plan_info['best_cost']:.3f}"
            )

        state = next_state
        final_reason = str(step_info["final_reason"])
        if done:
            break

    states_np = np.asarray(states, dtype=float)
    actions_np = np.asarray(actions, dtype=float)
    yaw_refs_np = np.asarray(yaw_refs, dtype=float)
    roll_refs_np = np.asarray(roll_refs, dtype=float)
    n = min(len(states_np), len(yaw_refs_np))
    yaw_error = _angle_error(states_np[:n, 6], yaw_refs_np[:n]) if n else np.asarray([0.0])
    roll_error = states_np[:n, 8] - roll_refs_np[:n] if n else np.asarray([0.0])

    final_yaw_error_deg = float(np.rad2deg(yaw_error[-1])) if n else 0.0
    max_abs_roll = float(np.max(np.abs(states_np[:, 8]))) if len(states_np) else 0.0
    max_abs_theta = float(np.max(np.abs(states_np[:, 0]))) if len(states_np) else 0.0
    max_abs_phi = float(np.max(np.abs(states_np[:, 4]))) if len(states_np) else 0.0
    max_abs_x = float(np.max(np.abs(states_np[:, 2]))) if len(states_np) else 0.0

    out = Path(args.out) if args.out else default_output_path(args, env)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        states=states_np,
        actions=actions_np,
        rewards=np.asarray(rewards, dtype=float),
        yaw_refs=yaw_refs_np,
        roll_refs=roll_refs_np,
        best_costs=np.asarray(best_costs, dtype=float),
        mean_costs=np.asarray(mean_costs, dtype=float),
        uncertainty_costs=np.asarray(uncertainty_costs, dtype=float),
        chance_penalties=np.asarray(chance_penalties, dtype=float),
        guide_costs=np.asarray(guide_costs, dtype=float),
        terminal_costs=np.asarray(terminal_costs, dtype=float),
        guide_action_norms=np.asarray([np.linalg.norm(info["guide_action"]) for info in infos], dtype=float),
        best_action_norms=np.asarray([np.linalg.norm(info["best_first_action"]) for info in infos], dtype=float),
        dt=env.p.dt,
        yaw_ref=ref.yaw_ref,
        v_ref=ref.v_ref,
        target_deg=args.target_deg,
        x_limit=ref.x_limit,
        uncertainty_weight=args.uncertainty_weight,
        chance_weight=args.chance_weight,
        guide_weight=args.guide_weight,
        terminal_weight=args.terminal_weight,
        k_sigma=args.k_sigma,
        horizon=args.horizon,
        candidates=args.candidates,
        seed=args.seed,
        T_limit=env.p.T_limit,
        Tp_limit=env.p.Tp_limit,
        final_reason=np.array([final_reason]),
        final_yaw_error_deg=np.array([final_yaw_error_deg]),
        max_abs_roll=np.array([max_abs_roll]),
        max_abs_roll_error=np.array([float(np.max(np.abs(roll_error))) if n else 0.0]),
        max_abs_theta=np.array([max_abs_theta]),
        max_abs_phi=np.array([max_abs_phi]),
        max_abs_x=np.array([max_abs_x]),
    )

    print(f"saved={out}")
    print(f"steps={len(states_np)}")
    print(f"final_reason={final_reason}")
    print(f"final_yaw_error_deg={final_yaw_error_deg:.3f}")
    print(f"max_abs_roll={max_abs_roll:.4f}")
    print(f"max_abs_theta={max_abs_theta:.4f}")
    print(f"max_abs_phi={max_abs_phi:.4f}")
    print(f"max_abs_x={max_abs_x:.4f}")


if __name__ == "__main__":
    main()
