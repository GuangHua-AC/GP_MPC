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


def l0_reference(args, step: int, dt: float) -> tuple[float, float]:
    t = step * dt
    if args.mode == "fixed":
        return float(args.L0_ref), 0.0
    if args.mode == "step":
        return (float(args.low) if t < args.switch_time else float(args.high)), 0.0
    if args.mode == "sine":
        mid = 0.5 * (args.low + args.high)
        amp = 0.5 * (args.high - args.low)
        omega = 2.0 * np.pi / max(args.period, 1e-6)
        return float(mid + amp * np.sin(omega * t)), float(amp * omega * np.cos(omega * t))
    raise ValueError(f"unknown height mode: {args.mode}")


def tracking_metrics(l0s: np.ndarray, l0_refs: np.ndarray, dt: float) -> dict[str, float]:
    n = min(len(l0s), len(l0_refs))
    if n == 0:
        return {
            "max_abs_L0_error": 0.0,
            "max_abs_L0_error_after_1s": 0.0,
            "rmse_L0_error": 0.0,
            "rmse_L0_error_after_1s": 0.0,
            "final_L0": 0.0,
            "final_L0_error": 0.0,
            "settling_time_2cm": np.nan,
        }
    err = np.asarray(l0s[:n] - l0_refs[:n], dtype=float)
    start = min(n - 1, int(np.ceil(1.0 / max(dt, 1e-9))))
    err_after = err[start:]

    # Approximate settling time: first time when the next 0.5 s window stays
    # within 2 cm. This is robust for fixed/step references and still useful as
    # a local tracking-quality marker for sine references.
    window = max(1, int(round(0.5 / max(dt, 1e-9))))
    settling_time = np.nan
    for i in range(n):
        j = min(n, i + window)
        if np.all(np.abs(err[i:j]) <= 0.02):
            settling_time = i * dt
            break

    return {
        "max_abs_L0_error": float(np.max(np.abs(err))),
        "max_abs_L0_error_after_1s": float(np.max(np.abs(err_after))) if len(err_after) else float(np.max(np.abs(err))),
        "rmse_L0_error": float(np.sqrt(np.mean(err**2))),
        "rmse_L0_error_after_1s": float(np.sqrt(np.mean(err_after**2))) if len(err_after) else float(np.sqrt(np.mean(err**2))),
        "final_L0": float(l0s[n - 1]),
        "final_L0_error": float(err[-1]),
        "settling_time_2cm": float(settling_time),
    }


def default_output_path(args, env: WheelLeggedEnv) -> Path:
    l0_tag = _tag(args.L0_ref)
    start_tag = _tag(args.L0_start)
    low_tag = _tag(args.low)
    high_tag = _tag(args.high)
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
        / "height"
        / "pmpc"
        / (
            f"height_pmpc_{args.mode}_L0ref{l0_tag}_L{low_tag}_to_{high_tag}_L0start{start_tag}_T{t_tag}_Tp{tp_tag}_"
            f"Uw{uw_tag}_Cw{cw_tag}_Gw{gw_tag}_K{ks_tag}_Tw{tw_tag}_seed{seed_tag}.npz"
        )
    )


def _missing_model_message(model_path: Path) -> str:
    return (
        f"Missing height GP model: {model_path}\n"
        "Please run:\n"
        "python scripts/common/02_collect_data.py --task height --episodes 120 --steps 600 --noise-scale 0.06\n"
        "python scripts/common/05_train_gp.py --task height --max-points 1500"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed", "step", "sine"], default="fixed")
    parser.add_argument("--low", type=float, default=0.30)
    parser.add_argument("--high", type=float, default=0.34)
    parser.add_argument("--period", type=float, default=4.0)
    parser.add_argument("--switch-time", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--uncertainty-weight", type=float, default=5.0)
    parser.add_argument("--chance-weight", type=float, default=20.0)
    parser.add_argument("--guide-weight", type=float, default=20.0)
    parser.add_argument("--terminal-weight", type=float, default=0.0)
    parser.add_argument("--k-sigma", type=float, default=2.0)
    parser.add_argument("--L0-ref", dest="L0_ref", type=float, default=0.34)
    parser.add_argument("--L0-start", dest="L0_start", type=float, default=0.30)
    parser.add_argument("--x-limit", type=float, default=2.0)
    parser.add_argument("--T-limit", type=float, default=1.2)
    parser.add_argument("--Tp-limit", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--random-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_dirs()
    model_path = task_output_subdir("height", "models") / "height_gp.joblib"
    if not model_path.exists():
        print(_missing_model_message(model_path))
        raise SystemExit(2)

    env = WheelLeggedEnv(task="height")
    env.p.max_steps = args.steps
    env.p.T_limit = float(args.T_limit)
    env.p.Tp_limit = float(args.Tp_limit)
    initial_ref, initial_dot_ref = l0_reference(args, 0, env.p.dt)
    ref = Reference(x_limit=args.x_limit, L0_ref=initial_ref, L0_dot_ref=initial_dot_ref)

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
        k_sigma=args.k_sigma,
        terminal_weight=args.terminal_weight,
        chance_enabled=("theta", "phi", "x", "L0"),
    )
    mpc = RiskAwareShootingMPC(env, model, cfg, WheelLeggedPDController(env))
    state = env.reset(L0_init=args.L0_start, ref=ref)

    states = []
    actions = []
    rewards = []
    l0s = []
    l0_refs = []
    l0_dot_refs = []
    best_costs = []
    mean_costs = []
    uncertainty_costs = []
    chance_penalties = []
    guide_costs = []
    terminal_costs = []
    infos = []
    final_reason = "not_done"

    for step in range(args.steps):
        ref_l0, ref_l0_dot = l0_reference(args, step, env.p.dt)
        ref = Reference(x_limit=args.x_limit, L0_ref=ref_l0, L0_dot_ref=ref_l0_dot)
        action, plan_info = mpc.plan(state, ref)
        next_state, reward, done, step_info = env.step(action, ref)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        l0s.append(env.leg.L0(state[10]))
        l0_refs.append(ref.L0_ref)
        l0_dot_refs.append(ref.L0_dot_ref)
        best_costs.append(plan_info["best_cost"])
        mean_costs.append(plan_info["mean_cost"])
        uncertainty_costs.append(plan_info["uncertainty_cost"])
        chance_penalties.append(plan_info["chance_penalty"])
        guide_costs.append(plan_info["guide_cost"])
        terminal_costs.append(plan_info["terminal_cost"])
        infos.append({**plan_info, **step_info})

        if step % 100 == 0:
            l0 = env.leg.L0(state[10])
            print(
                f"debug step={step} L0={l0:.4f} L0_err={l0 - ref.L0_ref:+.4f} "
                f"theta={state[0]:+.4f} phi={state[4]:+.4f} x={state[2]:+.4f} "
                f"best_cost={plan_info['best_cost']:.3f}"
            )

        state = next_state
        final_reason = str(step_info["final_reason"])
        if done:
            break

    states_np = np.asarray(states, dtype=float)
    actions_np = np.asarray(actions, dtype=float)
    l0s_np = np.asarray(l0s, dtype=float)
    l0_refs_np = np.asarray(l0_refs, dtype=float)
    n = min(len(l0s_np), len(l0_refs_np))
    l0_err = l0s_np[:n] - l0_refs_np[:n] if n else np.asarray([0.0])
    metrics = tracking_metrics(l0s_np, l0_refs_np, env.p.dt)
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
        L0s=l0s_np,
        L0_values=l0s_np,
        L0_refs=l0_refs_np,
        L0_dot_refs=np.asarray(l0_dot_refs, dtype=float),
        L0_errors=l0_err,
        best_costs=np.asarray(best_costs, dtype=float),
        mean_costs=np.asarray(mean_costs, dtype=float),
        uncertainty_costs=np.asarray(uncertainty_costs, dtype=float),
        chance_penalties=np.asarray(chance_penalties, dtype=float),
        guide_costs=np.asarray(guide_costs, dtype=float),
        terminal_costs=np.asarray(terminal_costs, dtype=float),
        guide_action_norms=np.asarray([np.linalg.norm(info["guide_action"]) for info in infos], dtype=float),
        best_action_norms=np.asarray([np.linalg.norm(info["best_first_action"]) for info in infos], dtype=float),
        dt=env.p.dt,
        mode=args.mode,
        L0_ref=args.L0_ref,
        L0_start=args.L0_start,
        low=args.low,
        high=args.high,
        period=args.period,
        switch_time=args.switch_time,
        x_limit=ref.x_limit,
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
        T_limit=env.p.T_limit,
        Tp_limit=env.p.Tp_limit,
        final_reason=np.array([final_reason]),
        final_L0=np.array([metrics["final_L0"]]),
        final_L0_error=np.array([metrics["final_L0_error"]]),
        max_abs_L0_error=np.array([metrics["max_abs_L0_error"]]),
        max_abs_L0_error_after_1s=np.array([metrics["max_abs_L0_error_after_1s"]]),
        rmse_L0_error=np.array([metrics["rmse_L0_error"]]),
        rmse_L0_error_after_1s=np.array([metrics["rmse_L0_error_after_1s"]]),
        settling_time_2cm=np.array([metrics["settling_time_2cm"]]),
        max_abs_theta=np.array([max_abs_theta]),
        max_abs_phi=np.array([max_abs_phi]),
        max_abs_x=np.array([max_abs_x]),
    )

    print(f"saved={out}")
    print(f"steps={len(states_np)}")
    print(f"final_reason={final_reason}")
    print(f"max_abs_L0_error={metrics['max_abs_L0_error']:.5f}")
    print(f"max_abs_L0_error_after_1s={metrics['max_abs_L0_error_after_1s']:.5f}")
    print(f"rmse_L0_error_after_1s={metrics['rmse_L0_error_after_1s']:.5f}")
    print(f"final_L0_error={metrics['final_L0_error']:.5f}")
    print(f"settling_time_2cm={metrics['settling_time_2cm']:.3f}")
    print(f"max_abs_theta={max_abs_theta:.4f}")
    print(f"max_abs_phi={max_abs_phi:.4f}")
    print(f"max_abs_x={max_abs_x:.4f}")


if __name__ == "__main__":
    main()
