from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from wheel_legged.controllers import PDGains, RandomShootingMPC, WheelLeggedPDController
from wheel_legged.controllers.mpc import MPCConfig
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv
from wheel_legged.models import NNDynamicsModel
from wheel_legged.utils.paths import ensure_dirs, task_output_subdir

from _common import controller_label, roll_turn_filename, save_roll_turn_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--target-deg", type=float, default=30.0)
    parser.add_argument("--roll-ref-deg", type=float, default=0.0)
    parser.add_argument("--roll0-deg", type=float, default=0.0)
    parser.add_argument("--v-ref", type=float, default=0.15)
    parser.add_argument("--theta0", type=float, default=0.03)
    parser.add_argument("--phi0", type=float, default=0.03)
    parser.add_argument("--x-limit", type=float, default=5.0)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--backend", default="torch", choices=["sklearn", "torch"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roll-centrifugal-ff-scale", type=float, default=0.0)
    args = parser.parse_args()

    ensure_dirs()
    env = WheelLeggedEnv(task="balance_turn_roll")
    env.p.max_steps = args.steps
    if args.backend == "torch":
        from wheel_legged.models.torch_dynamics import TorchDynamicsModel

        model = TorchDynamicsModel.load(
            task_output_subdir("balance_turn_roll", "models") / "balance_turn_roll_nn_torch.pt",
            device=args.device,
        )
    else:
        model = NNDynamicsModel.load(task_output_subdir("balance_turn_roll", "models") / "balance_turn_roll_nn.pt")

    ref = Reference(
        yaw_ref=np.deg2rad(args.target_deg),
        roll_ref=np.deg2rad(args.roll_ref_deg),
        v_ref=args.v_ref,
        x_limit=args.x_limit,
    )
    mpc = RandomShootingMPC(
        env,
        model,
        MPCConfig(horizon=args.horizon, candidates=args.candidates),
        WheelLeggedPDController(env, PDGains(roll_centrifugal_ff_scale=args.roll_centrifugal_ff_scale)),
    )
    state = env.reset(theta0=args.theta0, phi0=args.phi0, roll0=np.deg2rad(args.roll0_deg), ref=ref)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    yaw_refs: list[float] = []
    roll_refs: list[float] = []
    best_costs: list[float] = []
    final_reason = "not_done"

    for _ in range(args.steps):
        action, plan_info = mpc.plan(state, ref)
        next_state, reward, done, info = env.step(action, ref)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        yaw_refs.append(ref.yaw_ref)
        roll_refs.append(ref.roll_ref)
        best_costs.append(plan_info["best_cost"])
        state = next_state
        final_reason = info["final_reason"]
        if done:
            break

    out = task_output_subdir("balance_turn_roll", "mpc") / roll_turn_filename(
        "nn_mpc",
        args.target_deg,
        args.v_ref,
        args.roll0_deg,
        backend=args.backend,
    )
    metrics = save_roll_turn_result(
        out,
        states=np.asarray(states),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        yaw_refs=np.asarray(yaw_refs),
        roll_refs=np.asarray(roll_refs),
        dt=env.p.dt,
        target_deg=args.target_deg,
        v_ref=args.v_ref,
        roll0_deg=args.roll0_deg,
        controller=controller_label("nn_mpc"),
        final_reason=final_reason,
        horizon=args.horizon,
        candidates=args.candidates,
        best_costs=np.asarray(best_costs),
    )

    print("task=balance_turn_roll_nn_mpc")
    print(f"backend={args.backend} device={args.device}")
    print(f"steps={metrics['steps']} final_reason={metrics['final_reason']} success={metrics['success']}")
    print(f"max_abs_yaw_error_deg={np.rad2deg(metrics['max_abs_yaw_error_rad']):.3f}")
    print(f"max_abs_roll_error_deg={np.rad2deg(metrics['max_abs_roll_error_rad']):.3f}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
