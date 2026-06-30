from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

TERRAIN_ADAPTIVE_DIR = SCRIPTS_DIR / "terrain_adaptive"
if str(TERRAIN_ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(TERRAIN_ADAPTIVE_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from wheel_legged.dynamics.env import Reference, WheelLeggedEnv
from wheel_legged.models import GPDynamicsModel
from wheel_legged.pmpc import RiskAwareMPCConfig, RiskAwareShootingMPC
from wheel_legged.utils.paths import OUTPUT_DIR, task_output_subdir

from _common import AdaptiveGains, BlindTerrainCostEnv, OnlineTerrainAdaptiveController, terrain_trace_for_state


ADAPTIVE_OUTPUT_DIR = OUTPUT_DIR / "terrain_adaptive"


@dataclass
class SimpleAdaptiveGains:
    adaptive_gain: float
    adaptive_limit: float


class SimpleRollAdaptiveGuide(OnlineTerrainAdaptiveController):
    """Blind adaptive guide with a simple roll-to-leg-diff update law."""

    def __init__(self, env: WheelLeggedEnv, simple_gains: SimpleAdaptiveGains):
        super().__init__(env, adaptive_gains=AdaptiveGains(adapt_rate=0.0, adapt_damping=0.0, leak=0.0))
        self.simple_gains = simple_gains

    def _update_leg_diff_ref(self, roll: float, roll_dot: float) -> None:
        del roll_dot
        value = -self.simple_gains.adaptive_gain * roll
        self.leg_diff_ref = float(np.clip(value, -self.simple_gains.adaptive_limit, self.simple_gains.adaptive_limit))
        self.last_leg_diff_ref = self.leg_diff_ref


def _tag(value: float | int | str) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m") if isinstance(value, (float, int)) else str(value)


def default_output_path(args) -> Path:
    return ADAPTIVE_OUTPUT_DIR / "pmpc" / (
        f"terrain_adaptive_pmpc_{args.terrain_mode}_h{_tag(args.obstacle_height)}_x{_tag(args.obstacle_start)}_"
        f"len{_tag(args.obstacle_length)}_Ag{_tag(args.adaptive_gain)}_Alim{_tag(args.adaptive_limit)}_"
        f"Uw{_tag(args.uncertainty_weight)}_Cw{_tag(args.chance_weight)}_Gw{_tag(args.guide_weight)}_"
        f"N{_tag(args.noise_scale)}_Rf{_tag(args.random_fraction)}_seed{_tag(args.seed)}_gp_pmpc.npz"
    )


def find_model() -> tuple[Path, str]:
    candidates = [
        (ADAPTIVE_OUTPUT_DIR / "models" / "terrain_adaptive_gp.joblib", "terrain_adaptive_gp"),
        (OUTPUT_DIR / "terrain" / "models" / "terrain_adaptive_gp.joblib", "terrain_adaptive_gp"),
        (task_output_subdir("terrain", "models") / "terrain_gp.joblib", "terrain_gp_fallback"),
    ]
    for path, source in candidates:
        if path.exists():
            return path, source
    raise FileNotFoundError(
        "Missing terrain GP model. Please run:\n"
        "python scripts/common/02_collect_data.py --task terrain --episodes 120 --steps 600 --noise-scale 0.06\n"
        "python scripts/common/05_train_gp.py --task terrain --max-points 1500"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--uncertainty-weight", type=float, default=5.0)
    parser.add_argument("--chance-weight", type=float, default=200.0)
    parser.add_argument("--guide-weight", type=float, default=50.0)
    parser.add_argument("--terminal-weight", type=float, default=0.0)
    parser.add_argument("--k-sigma", type=float, default=2.0)
    parser.add_argument("--terrain-mode", choices=["left_obstacle", "right_obstacle", "sine"], default="left_obstacle")
    parser.add_argument("--obstacle-height", type=float, default=0.04)
    parser.add_argument("--obstacle-start", type=float, default=1.0)
    parser.add_argument("--obstacle-length", type=float, default=0.5)
    parser.add_argument("--adaptive-gain", type=float, default=0.5)
    parser.add_argument("--adaptive-limit", type=float, default=0.08)
    parser.add_argument("--v-ref", type=float, default=0.15)
    parser.add_argument("--L0-ref", type=float, default=0.32)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--T-limit", type=float, default=1.2)
    parser.add_argument("--Tp-limit", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--random-fraction", type=float, default=0.0)
    parser.add_argument("--theta0", type=float, default=0.03)
    parser.add_argument("--phi0", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ADAPTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("pmpc", "metrics", "figures", "videos"):
        (ADAPTIVE_OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    model_path, model_source = find_model()
    env = WheelLeggedEnv(task="terrain", terrain_mode=args.terrain_mode)
    env.p.max_steps = args.steps
    env.p.T_limit = args.T_limit
    env.p.Tp_limit = args.Tp_limit
    env.p.obstacle_height = args.obstacle_height
    env.p.obstacle_start = args.obstacle_start
    env.p.obstacle_length = args.obstacle_length

    planning_env = BlindTerrainCostEnv(task="terrain", params=env.p, terrain_mode=args.terrain_mode)
    planning_env.terrain_known_to_controller = False
    planning_env.p.max_steps = args.steps
    planning_env.p.T_limit = args.T_limit
    planning_env.p.Tp_limit = args.Tp_limit
    planning_env.p.obstacle_height = args.obstacle_height
    planning_env.p.obstacle_start = args.obstacle_start
    planning_env.p.obstacle_length = args.obstacle_length

    ref = Reference(x_limit=args.x_limit, v_ref=args.v_ref, L0_ref=args.L0_ref)
    guide = SimpleRollAdaptiveGuide(env, SimpleAdaptiveGains(args.adaptive_gain, args.adaptive_limit))
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
        guide_action_weights=np.asarray([1.0, 1.0, 1.0, 5.0, 2.0, 5.0], dtype=float),
        k_sigma=args.k_sigma,
        terminal_weight=args.terminal_weight,
        chance_enabled=("theta", "phi", "x", "roll", "leg_diff"),
    )
    mpc = RiskAwareShootingMPC(planning_env, model, cfg, guide)
    state = env.reset(theta0=args.theta0, phi0=args.phi0, L0_init=args.L0_ref, ref=ref)
    planning_env.ref = ref
    guide.reset()

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    left_heights: list[float] = []
    right_heights: list[float] = []
    terrain_diffs: list[float] = []
    support_rolls: list[float] = []
    leg_diff_adapt_values: list[float] = []
    best_costs: list[float] = []
    chance_penalties: list[float] = []
    guide_costs: list[float] = []
    plan_times_sec: list[float] = []
    final_reason = "not_done"
    rollout_start = time.perf_counter()

    for step in range(args.steps):
        plan_start = time.perf_counter()
        action, plan_info = mpc.plan(state, ref)
        plan_times_sec.append(float(time.perf_counter() - plan_start))
        left_h, right_h, terrain_diff, support_roll = terrain_trace_for_state(env, state)
        next_state, reward, done, info = env.step(action, ref)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        left_heights.append(left_h)
        right_heights.append(right_h)
        terrain_diffs.append(terrain_diff)
        support_rolls.append(support_roll)
        leg_diff_adapt_values.append(float(guide.last_leg_diff_ref))
        best_costs.append(float(plan_info["best_cost"]))
        chance_penalties.append(float(plan_info["chance_penalty"]))
        guide_costs.append(float(plan_info["guide_cost"]))

        if step % 100 == 0:
            print(
                f"debug step={step} theta={state[0]:+.4f} phi={state[4]:+.4f} roll={state[8]:+.4f} "
                f"x={state[2]:+.4f} adapt={guide.last_leg_diff_ref:+.4f} best_cost={plan_info['best_cost']:.3f}"
            )

        state = next_state
        final_reason = str(info["final_reason"])
        if done:
            break

    states_np = np.asarray(states, dtype=float)
    actions_np = np.asarray(actions, dtype=float)
    support_np = np.asarray(support_rolls, dtype=float)
    leg_adapt_np = np.asarray(leg_diff_adapt_values, dtype=float)
    plan_times_np = np.asarray(plan_times_sec, dtype=float)
    action_norms = np.linalg.norm(actions_np, axis=1) if len(actions_np) else np.asarray([0.0])
    total_runtime_sec = float(time.perf_counter() - rollout_start)
    max_abs_roll = float(np.max(np.abs(states_np[:, 8]))) if len(states_np) else 0.0
    max_abs_support_roll = float(np.max(np.abs(support_np))) if len(support_np) else 0.0
    max_abs_theta = float(np.max(np.abs(states_np[:, 0]))) if len(states_np) else 0.0
    max_abs_phi = float(np.max(np.abs(states_np[:, 4]))) if len(states_np) else 0.0
    max_abs_x = float(np.max(np.abs(states_np[:, 2]))) if len(states_np) else 0.0
    max_abs_leg_diff = float(np.max(np.abs(states_np[:, 12]))) if len(states_np) else 0.0

    out = Path(args.out) if args.out else default_output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        states=states_np,
        actions=actions_np,
        rewards=np.asarray(rewards, dtype=float),
        dt=env.p.dt,
        terrain_mode=args.terrain_mode,
        obstacle_height=args.obstacle_height,
        obstacle_start=args.obstacle_start,
        obstacle_length=args.obstacle_length,
        adaptive_gain=args.adaptive_gain,
        adaptive_limit=args.adaptive_limit,
        leg_diff_adapt_values=leg_adapt_np,
        leg_diff_values=states_np[:, 12] if len(states_np) else np.asarray([], dtype=float),
        roll_values=states_np[:, 8] if len(states_np) else np.asarray([], dtype=float),
        support_roll_values=support_np,
        support_rolls=support_np,
        left_heights=np.asarray(left_heights, dtype=float),
        right_heights=np.asarray(right_heights, dtype=float),
        terrain_diffs=np.asarray(terrain_diffs, dtype=float),
        x_limit=args.x_limit,
        uncertainty_weight=args.uncertainty_weight,
        chance_weight=args.chance_weight,
        guide_weight=args.guide_weight,
        terminal_weight=args.terminal_weight,
        k_sigma=args.k_sigma,
        noise_scale=args.noise_scale,
        random_fraction=args.random_fraction,
        horizon=args.horizon,
        candidates=args.candidates,
        seed=args.seed,
        model_source=model_source,
        terrain_known_to_controller=np.array([False]),
        final_reason=np.array([final_reason]),
        best_costs=np.asarray(best_costs, dtype=float),
        chance_penalties=np.asarray(chance_penalties, dtype=float),
        guide_costs=np.asarray(guide_costs, dtype=float),
        plan_times_sec=plan_times_np,
        mean_plan_time_sec=np.array([float(np.mean(plan_times_np)) if len(plan_times_np) else 0.0]),
        max_plan_time_sec=np.array([float(np.max(plan_times_np)) if len(plan_times_np) else 0.0]),
        total_runtime_sec=np.array([total_runtime_sec]),
        mean_action_norm=np.array([float(np.mean(action_norms))]),
        max_action_norm=np.array([float(np.max(action_norms))]),
    )

    print(f"model_source={model_source}")
    print("terrain_known_to_controller=False")
    print(f"saved={out}")
    print(f"steps={len(states_np)}")
    print(f"final_reason={final_reason}")
    print(f"max_abs_roll={max_abs_roll:.5f} rad ({np.rad2deg(max_abs_roll):.3f} deg)")
    print(f"max_abs_support_roll={max_abs_support_roll:.5f} rad ({np.rad2deg(max_abs_support_roll):.3f} deg)")
    print(f"max_abs_theta={max_abs_theta:.5f}")
    print(f"max_abs_phi={max_abs_phi:.5f}")
    print(f"max_abs_x={max_abs_x:.5f}")
    print(f"max_abs_leg_diff={max_abs_leg_diff:.5f}")
    print(f"mean_plan_time_sec={float(np.mean(plan_times_np)) if len(plan_times_np) else 0.0:.5f}")
    print(f"total_runtime_sec={total_runtime_sec:.3f}")


if __name__ == "__main__":
    main()
