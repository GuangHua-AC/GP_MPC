from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from wheel_legged.controllers import RandomShootingMPC
from wheel_legged.controllers.mpc import MPCConfig
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv
from wheel_legged.models import NNDynamicsModel
from wheel_legged.utils.paths import task_output_subdir

from _common import (
    ADAPTIVE_DIR,
    AdaptiveGains,
    BlindTerrainCostEnv,
    OnlineTerrainAdaptiveController,
    adaptive_filename,
    controller_label,
    ensure_adaptive_dirs,
    save_adaptive_result,
    terrain_trace_for_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sklearn", "torch"], default="torch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--terrain-mode", choices=["left_obstacle", "right_obstacle", "sine"], default="left_obstacle")
    parser.add_argument("--v-ref", type=float, default=0.15)
    parser.add_argument("--L0-ref", type=float, default=0.32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--random-fraction", type=float, default=0.0)
    parser.add_argument("--mpc-blend", type=float, default=0.20)
    parser.add_argument("--theta0", type=float, default=0.03)
    parser.add_argument("--phi0", type=float, default=0.03)
    parser.add_argument("--x-limit", type=float, default=5.0)
    parser.add_argument("--adapt-rate", type=float, default=4.0)
    parser.add_argument("--adapt-damping", type=float, default=0.10)
    parser.add_argument("--adapt-leak", type=float, default=0.02)
    args = parser.parse_args()

    ensure_adaptive_dirs()
    env = WheelLeggedEnv(task="terrain", terrain_mode=args.terrain_mode)
    env.p.max_steps = args.steps
    planning_env = BlindTerrainCostEnv(task="terrain", params=env.p, terrain_mode=args.terrain_mode)
    planning_env.p.max_steps = args.steps

    if args.backend == "torch":
        from wheel_legged.models.torch_dynamics import TorchDynamicsModel

        model = TorchDynamicsModel.load(task_output_subdir("terrain", "models") / "terrain_nn_torch.pt", device=args.device)
    else:
        model = NNDynamicsModel.load(task_output_subdir("terrain", "models") / "terrain_nn.pt")

    guide = OnlineTerrainAdaptiveController(
        env,
        adaptive_gains=AdaptiveGains(
            adapt_rate=args.adapt_rate,
            adapt_damping=args.adapt_damping,
            leak=args.adapt_leak,
        ),
    )
    mpc = RandomShootingMPC(
        planning_env,
        model,
        MPCConfig(
            horizon=args.horizon,
            candidates=args.candidates,
            noise_scale=args.noise_scale,
            random_fraction=args.random_fraction,
        ),
        guide,
    )
    ref = Reference(x_limit=args.x_limit, v_ref=args.v_ref, L0_ref=args.L0_ref)
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
    leg_diff_refs: list[float] = []
    best_costs: list[float] = []
    final_reason = "not_done"

    for _ in range(args.steps):
        mpc_action, plan_info = mpc.plan(state, ref)
        guide_action = guide.last_action if guide.last_action is not None else guide.act(state, ref)
        action = env._clip_action(guide_action + args.mpc_blend * (mpc_action - guide_action))
        left_h, right_h, terrain_diff, support_roll = terrain_trace_for_state(env, state)
        next_state, reward, done, info = env.step(action, ref)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        left_heights.append(left_h)
        right_heights.append(right_h)
        terrain_diffs.append(terrain_diff)
        support_rolls.append(support_roll)
        leg_diff_refs.append(guide.last_leg_diff_ref)
        best_costs.append(float(plan_info.get("best_cost", 0.0)))
        state = next_state
        final_reason = info["final_reason"]
        if done:
            break

    out = ADAPTIVE_DIR / "mpc" / adaptive_filename("adaptive_nn_mpc", args.terrain_mode, args.v_ref, backend=args.backend)
    metrics = save_adaptive_result(
        out,
        states=np.asarray(states),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        left_heights=np.asarray(left_heights),
        right_heights=np.asarray(right_heights),
        terrain_diffs=np.asarray(terrain_diffs),
        support_rolls=np.asarray(support_rolls),
        leg_diff_refs=np.asarray(leg_diff_refs),
        dt=env.p.dt,
        terrain_mode=args.terrain_mode,
        v_ref=args.v_ref,
        controller=controller_label("adaptive_nn_mpc"),
        final_reason=final_reason,
        horizon=args.horizon,
        candidates=args.candidates,
        mpc_blend=args.mpc_blend,
        best_costs=np.asarray(best_costs),
    )
    print("task=terrain_adaptive_nn_mpc")
    print("terrain_known_to_controller=False")
    print(f"backend={args.backend} device={args.device}")
    print(f"terrain_mode={args.terrain_mode} v_ref={args.v_ref}")
    print(f"horizon={args.horizon} candidates={args.candidates} noise_scale={args.noise_scale} random_fraction={args.random_fraction} mpc_blend={args.mpc_blend}")
    print(f"steps={metrics['steps']} final_reason={metrics['final_reason']} success={metrics['success']}")
    print(f"max_abs_roll_deg={np.rad2deg(metrics['max_abs_roll_rad']):.3f}")
    print(f"max_abs_support_roll_deg={np.rad2deg(metrics['max_abs_support_roll_rad']):.3f}")
    print(f"max_abs_leg_diff_m={metrics['max_abs_leg_diff_m']:.4f}")
    print(f"terrain_comp_rmse_m={metrics['terrain_comp_rmse_m']:.5f}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
