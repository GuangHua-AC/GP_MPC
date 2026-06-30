from __future__ import annotations

import argparse
import sys
import time
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


def _tag(value: float | int | str) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m") if isinstance(value, (float, int)) else str(value)


def _terrain_refs(info: dict, state: np.ndarray) -> tuple[float, float, float, float]:
    terrain_diff = float(info["terrain_diff"])
    leg_diff_ref = -terrain_diff
    leg_diff_value = float(state[12])
    leg_diff_error = leg_diff_value - leg_diff_ref
    support_roll = float(info["support_roll"])
    return leg_diff_ref, leg_diff_value, leg_diff_error, support_roll


def _parse_action_weights(text: str) -> np.ndarray:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("--guide-action-weights must contain six comma-separated numbers")
    return np.asarray(values, dtype=float)


def default_output_path(args, env: WheelLeggedEnv) -> Path:
    return (
        task_output_subdir("terrain", "pmpc")
        / (
            f"terrain_pmpc_{args.terrain_mode}_h{_tag(args.obstacle_height)}_x{_tag(args.obstacle_start)}_"
            f"len{_tag(args.obstacle_length)}_T{_tag(env.p.T_limit)}_Tp{_tag(env.p.Tp_limit)}_"
            f"Uw{_tag(args.uncertainty_weight)}_Cw{_tag(args.chance_weight)}_Gw{_tag(args.guide_weight)}_"
            f"N{_tag(args.noise_scale)}_Rf{_tag(args.random_fraction)}_K{_tag(args.k_sigma)}_"
            f"Tw{_tag(args.terminal_weight)}_seed{_tag(args.seed)}_gp_pmpc.npz"
        )
    )


def missing_model_message(model_path: Path) -> str:
    return (
        f"Missing terrain GP model: {model_path}\n"
        "Please run:\n"
        "python scripts/common/02_collect_data.py --task terrain --episodes 120 --steps 600 --noise-scale 0.06\n"
        "python scripts/common/05_train_gp.py --task terrain --max-points 1500"
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
    parser.add_argument("--terrain-mode", choices=["left_obstacle", "right_obstacle", "sine"], default="left_obstacle")
    parser.add_argument("--obstacle-height", type=float, default=0.04)
    parser.add_argument("--obstacle-start", type=float, default=1.0)
    parser.add_argument("--obstacle-length", type=float, default=0.5)
    parser.add_argument("--v-ref", type=float, default=0.15)
    parser.add_argument("--L0-ref", type=float, default=0.32)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--T-limit", type=float, default=1.2)
    parser.add_argument("--Tp-limit", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--random-fraction", type=float, default=0.0)
    parser.add_argument("--guide-action-weights", type=_parse_action_weights, default=_parse_action_weights("1,1,1,5,2,5"))
    parser.add_argument("--theta0", type=float, default=0.03)
    parser.add_argument("--phi0", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_dirs()
    model_path = task_output_subdir("terrain", "models") / "terrain_gp.joblib"
    if not model_path.exists():
        print(missing_model_message(model_path))
        raise SystemExit(2)

    env = WheelLeggedEnv(task="terrain", terrain_mode=args.terrain_mode)
    env.p.max_steps = args.steps
    env.p.T_limit = float(args.T_limit)
    env.p.Tp_limit = float(args.Tp_limit)
    env.p.obstacle_height = float(args.obstacle_height)
    env.p.obstacle_start = float(args.obstacle_start)
    env.p.obstacle_length = float(args.obstacle_length)
    ref = Reference(x_limit=args.x_limit, v_ref=args.v_ref, L0_ref=args.L0_ref)

    model = GPDynamicsModel.load(model_path)
    cfg = RiskAwareMPCConfig(
        horizon=args.horizon,
        candidates=args.candidates,
        seed=args.seed,
        noise_scale=args.noise_scale,
        random_fraction=args.random_fraction,
        uncertainty_weight=args.uncertainty_weight,
        chance_weight=args.chance_weight,
        guide_weight=args.guide_weight,
        guide_action_weights=args.guide_action_weights,
        k_sigma=args.k_sigma,
        terminal_weight=args.terminal_weight,
        chance_enabled=("theta", "phi", "x", "roll", "leg_diff"),
    )
    mpc = RiskAwareShootingMPC(env, model, cfg, WheelLeggedPDController(env))
    state = env.reset(theta0=args.theta0, phi0=args.phi0, L0_init=args.L0_ref, ref=ref)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    left_heights: list[float] = []
    right_heights: list[float] = []
    terrain_diffs: list[float] = []
    leg_diff_refs: list[float] = []
    leg_diff_values: list[float] = []
    leg_diff_errors: list[float] = []
    roll_values: list[float] = []
    support_rolls: list[float] = []
    best_costs: list[float] = []
    mean_costs: list[float] = []
    uncertainty_costs: list[float] = []
    chance_penalties: list[float] = []
    guide_costs: list[float] = []
    terminal_costs: list[float] = []
    tracking_costs: list[float] = []
    guide_action_norms: list[float] = []
    best_action_norms: list[float] = []
    plan_times_sec: list[float] = []
    final_reason = "not_done"
    rollout_start = time.perf_counter()

    for step in range(args.steps):
        plan_start = time.perf_counter()
        action, plan_info = mpc.plan(state, ref)
        plan_time_sec = time.perf_counter() - plan_start
        pre_info = env.info(state, "not_done")
        leg_ref, leg_val, leg_err, support_roll = _terrain_refs(pre_info, state)
        next_state, reward, done, step_info = env.step(action, ref)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        left_heights.append(float(pre_info["left_ground_height"]))
        right_heights.append(float(pre_info["right_ground_height"]))
        terrain_diffs.append(float(pre_info["terrain_diff"]))
        leg_diff_refs.append(leg_ref)
        leg_diff_values.append(leg_val)
        leg_diff_errors.append(leg_err)
        roll_values.append(float(state[8]))
        support_rolls.append(support_roll)
        best_costs.append(float(plan_info["best_cost"]))
        mean_costs.append(float(plan_info["mean_cost"]))
        tracking_costs.append(float(plan_info["best_cost_tracking"]))
        uncertainty_costs.append(float(plan_info["uncertainty_cost"]))
        chance_penalties.append(float(plan_info["chance_penalty"]))
        guide_costs.append(float(plan_info["guide_cost"]))
        terminal_costs.append(float(plan_info["terminal_cost"]))
        guide_action_norms.append(float(np.linalg.norm(plan_info["guide_action"])))
        best_action_norms.append(float(np.linalg.norm(plan_info["best_first_action"])))
        plan_times_sec.append(float(plan_time_sec))

        if step % 100 == 0:
            print(
                f"debug step={step} theta={state[0]:+.4f} phi={state[4]:+.4f} roll={state[8]:+.4f} "
                f"x={state[2]:+.4f} leg_err={leg_err:+.4f} best_cost={plan_info['best_cost']:.3f}"
            )

        state = next_state
        final_reason = str(step_info["final_reason"])
        if done:
            break

    states_np = np.asarray(states, dtype=float)
    actions_np = np.asarray(actions, dtype=float)
    action_norms = np.linalg.norm(actions_np, axis=1) if len(actions_np) else np.asarray([0.0])
    plan_times_np = np.asarray(plan_times_sec, dtype=float)
    total_runtime_sec = float(time.perf_counter() - rollout_start)
    leg_err_np = np.asarray(leg_diff_errors, dtype=float)
    roll_np = np.asarray(roll_values, dtype=float)
    support_np = np.asarray(support_rolls, dtype=float)
    max_abs_leg_diff_error = float(np.max(np.abs(leg_err_np))) if len(leg_err_np) else 0.0
    max_abs_roll = float(np.max(np.abs(roll_np))) if len(roll_np) else 0.0
    max_abs_support_roll = float(np.max(np.abs(support_np))) if len(support_np) else 0.0
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
        left_heights=np.asarray(left_heights, dtype=float),
        right_heights=np.asarray(right_heights, dtype=float),
        terrain_diffs=np.asarray(terrain_diffs, dtype=float),
        leg_diff_refs=np.asarray(leg_diff_refs, dtype=float),
        leg_diff_values=np.asarray(leg_diff_values, dtype=float),
        leg_diff_errors=leg_err_np,
        roll_values=roll_np,
        support_roll_values=support_np,
        support_rolls=support_np,
        best_costs=np.asarray(best_costs, dtype=float),
        mean_costs=np.asarray(mean_costs, dtype=float),
        best_cost_tracking=np.asarray(tracking_costs, dtype=float),
        uncertainty_costs=np.asarray(uncertainty_costs, dtype=float),
        best_cost_uncertainty=np.asarray(uncertainty_costs, dtype=float),
        chance_penalties=np.asarray(chance_penalties, dtype=float),
        best_cost_chance=np.asarray(chance_penalties, dtype=float),
        guide_costs=np.asarray(guide_costs, dtype=float),
        best_cost_guide=np.asarray(guide_costs, dtype=float),
        terminal_costs=np.asarray(terminal_costs, dtype=float),
        best_cost_terminal=np.asarray(terminal_costs, dtype=float),
        guide_action_norms=np.asarray(guide_action_norms, dtype=float),
        best_action_norms=np.asarray(best_action_norms, dtype=float),
        plan_times_sec=plan_times_np,
        mean_plan_time_sec=np.array([float(np.mean(plan_times_np)) if len(plan_times_np) else 0.0]),
        max_plan_time_sec=np.array([float(np.max(plan_times_np)) if len(plan_times_np) else 0.0]),
        total_runtime_sec=np.array([total_runtime_sec]),
        dt=env.p.dt,
        terrain_mode=args.terrain_mode,
        obstacle_height=args.obstacle_height,
        obstacle_start=args.obstacle_start,
        obstacle_length=args.obstacle_length,
        x_limit=args.x_limit,
        v_ref=args.v_ref,
        L0_ref=args.L0_ref,
        uncertainty_weight=args.uncertainty_weight,
        chance_weight=args.chance_weight,
        guide_weight=args.guide_weight,
        terminal_weight=args.terminal_weight,
        k_sigma=args.k_sigma,
        horizon=args.horizon,
        candidates=args.candidates,
        seed=args.seed,
        noise_scale=args.noise_scale,
        random_fraction=args.random_fraction,
        guide_action_weights=args.guide_action_weights,
        T_limit=env.p.T_limit,
        Tp_limit=env.p.Tp_limit,
        final_reason=np.array([final_reason]),
        max_abs_leg_diff_error=np.array([max_abs_leg_diff_error]),
        max_abs_roll=np.array([max_abs_roll]),
        max_abs_support_roll=np.array([max_abs_support_roll]),
        max_abs_theta=np.array([max_abs_theta]),
        max_abs_phi=np.array([max_abs_phi]),
        max_abs_x=np.array([max_abs_x]),
        mean_action_norm=np.array([float(np.mean(action_norms))]),
        max_action_norm=np.array([float(np.max(action_norms))]),
    )

    print(f"saved={out}")
    print(f"steps={len(states_np)}")
    print(f"final_reason={final_reason}")
    print(f"max_abs_leg_diff_error={max_abs_leg_diff_error:.5f}")
    print(f"max_abs_roll={max_abs_roll:.5f} rad ({np.rad2deg(max_abs_roll):.3f} deg)")
    print(f"max_abs_support_roll={max_abs_support_roll:.5f} rad ({np.rad2deg(max_abs_support_roll):.3f} deg)")
    print(f"max_abs_theta={max_abs_theta:.5f}")
    print(f"max_abs_phi={max_abs_phi:.5f}")
    print(f"max_abs_x={max_abs_x:.5f}")
    print(f"mean_plan_time_sec={float(np.mean(plan_times_np)) if len(plan_times_np) else 0.0:.5f}")
    print(f"max_plan_time_sec={float(np.max(plan_times_np)) if len(plan_times_np) else 0.0:.5f}")
    print(f"total_runtime_sec={total_runtime_sec:.3f}")


if __name__ == "__main__":
    main()
