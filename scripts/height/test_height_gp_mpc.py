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

from _common import controller_label, height_filename, initial_l0, l0_reference, save_height_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["step", "step_cycle", "down", "sine", "fixed"], default="sine")
    parser.add_argument("--low", type=float, default=0.28)
    parser.add_argument("--high", type=float, default=0.36)
    parser.add_argument("--switch-time", type=float, default=1.5)
    parser.add_argument("--sine-freq", type=float, default=0.30)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--uncertainty-weight", type=float, default=5.0)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--random-fraction", type=float, default=0.0)
    parser.add_argument("--theta0", type=float, default=0.03)
    parser.add_argument("--phi0", type=float, default=0.03)
    parser.add_argument("--v-ref", type=float, default=0.15)
    parser.add_argument("--x-limit", type=float, default=2.0)
    args = parser.parse_args()

    ensure_dirs()
    env = WheelLeggedEnv(task="height")
    env.p.max_steps = args.steps
    model = GPDynamicsModel.load(task_output_subdir("height", "models") / "height_gp.joblib")
    mpc = RandomShootingMPC(
        env,
        model,
        MPCConfig(
            horizon=args.horizon,
            candidates=args.candidates,
            uncertainty_weight=args.uncertainty_weight,
            noise_scale=args.noise_scale,
            random_fraction=args.random_fraction,
        ),
        WheelLeggedPDController(env),
    )
    mid = 0.5 * (args.low + args.high)
    l0_init = initial_l0(args.mode, args.low, args.high, mid)
    l0_ref, l0_dot_ref = l0_reference(
        args.mode,
        0,
        env.p.dt,
        low=args.low,
        high=args.high,
        switch_time=args.switch_time,
        sine_freq=args.sine_freq,
    )
    ref = Reference(x_limit=args.x_limit, v_ref=args.v_ref, L0_ref=l0_ref, L0_dot_ref=l0_dot_ref)
    state = env.reset(theta0=args.theta0, phi0=args.phi0, L0_init=l0_init, ref=ref)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    l0s: list[float] = []
    l0_refs: list[float] = []
    l0_dot_refs: list[float] = []
    best_costs: list[float] = []
    uncertainty_costs: list[float] = []
    final_reason = "not_done"

    for step in range(args.steps):
        l0_ref, l0_dot_ref = l0_reference(
            args.mode,
            step,
            env.p.dt,
            low=args.low,
            high=args.high,
            switch_time=args.switch_time,
            sine_freq=args.sine_freq,
        )
        ref = Reference(x_limit=args.x_limit, v_ref=args.v_ref, L0_ref=l0_ref, L0_dot_ref=l0_dot_ref)
        action, plan_info = mpc.plan(state, ref)
        next_state, reward, done, step_info = env.step(action, ref)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        l0s.append(env.leg.L0(state[10]))
        l0_refs.append(l0_ref)
        l0_dot_refs.append(l0_dot_ref)
        best_costs.append(float(plan_info.get("best_cost", 0.0)))
        uncertainty_costs.append(float(plan_info.get("uncertainty_cost", 0.0)))
        state = next_state
        final_reason = step_info["final_reason"]
        if done:
            break

    out = task_output_subdir("height", "mpc") / height_filename("gp_mpc", args.mode, args.low, args.high, args.v_ref)
    metrics = save_height_result(
        out,
        states=np.asarray(states),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        l0s=np.asarray(l0s),
        l0_refs=np.asarray(l0_refs),
        l0_dot_refs=np.asarray(l0_dot_refs),
        dt=env.p.dt,
        mode=args.mode,
        low=args.low,
        high=args.high,
        v_ref=args.v_ref,
        controller=controller_label("gp_mpc"),
        final_reason=final_reason,
        horizon=args.horizon,
        candidates=args.candidates,
        uncertainty_weight=args.uncertainty_weight,
        best_costs=np.asarray(best_costs),
        uncertainty_costs=np.asarray(uncertainty_costs),
    )
    print("task=height_gp_mpc")
    print(f"mode={args.mode} low={args.low} high={args.high} v_ref={args.v_ref}")
    print(f"horizon={args.horizon} candidates={args.candidates} noise_scale={args.noise_scale} random_fraction={args.random_fraction}")
    print(f"steps={metrics['steps']} final_reason={metrics['final_reason']} success={metrics['success']}")
    print(f"l0_rmse_m={metrics['l0_rmse_m']:.5f} max_abs_l0_error_m={metrics['max_abs_l0_error_m']:.5f}")
    print(f"final_l0_error_m={metrics['final_l0_error_m']:.5f}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
