from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np

from wheel_legged.dynamics.env import Reference, WheelLeggedEnv

from _common import PANORAMA_DIR, ensure_panorama_dirs
from terrain_adaptive._common import OnlineTerrainAdaptiveController, terrain_trace_for_state


PHASES = ("balance_walk", "obstacle_turn", "height", "adaptive_terrain")


def phase_for_time(t: float) -> str:
    if t < 3.0:
        return "balance_walk"
    if t < 9.0:
        return "obstacle_turn"
    if t < 16.0:
        return "height"
    return "adaptive_terrain"


def reference_for_phase(phase: str, t: float, x_limit: float) -> Reference:
    if phase == "balance_walk":
        return Reference(x_limit=x_limit, v_ref=0.18, yaw_ref=0.0, L0_ref=0.27)
    if phase == "obstacle_turn":
        local_t = max(0.0, t - 3.0)
        yaw_ref = np.deg2rad(35.0) if local_t < 2.7 else 0.0
        return Reference(x_limit=x_limit, v_ref=0.25, yaw_ref=float(yaw_ref), L0_ref=0.27)
    if phase == "height":
        local_t = max(0.0, t - 9.0)
        if local_t < 2.3:
            L0_ref = 0.36
        elif local_t < 4.6:
            L0_ref = 0.235
        else:
            L0_ref = 0.27
        return Reference(x_limit=x_limit, v_ref=0.25, yaw_ref=0.0, L0_ref=float(L0_ref), L0_dot_ref=0.0)
    return Reference(x_limit=x_limit, v_ref=0.25, yaw_ref=0.0, L0_ref=0.27)


def external_push_for_time(t: float, force: float) -> float:
    if 0.80 <= t <= 0.95:
        return float(force)
    return 0.0


def integrate_world_path(states: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    world_x = np.zeros(len(states), dtype=float)
    world_y = np.zeros(len(states), dtype=float)
    for i in range(1, len(states)):
        ds = float(states[i - 1, 3] * dt)
        yaw = float(states[i - 1, 6])
        world_x[i] = world_x[i - 1] + ds * np.cos(yaw)
        world_y[i] = world_y[i - 1] + ds * np.sin(yaw)
    return world_x, world_y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5200)
    parser.add_argument("--push-force", type=float, default=10.0)
    parser.add_argument("--theta0", type=float, default=0.03)
    parser.add_argument("--phi0", type=float, default=0.03)
    parser.add_argument("--L0-init", type=float, default=0.27)
    parser.add_argument("--x-limit", type=float, default=8.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ensure_panorama_dirs()
    env = WheelLeggedEnv(task="terrain", terrain_mode="panorama")
    env.p.max_steps = args.steps
    controller = OnlineTerrainAdaptiveController(env)
    ref0 = reference_for_phase("balance_walk", 0.0, args.x_limit)
    state = env.reset(theta0=args.theta0, phi0=args.phi0, L0_init=args.L0_init, ref=ref0)
    controller.reset()

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    phase_ids: list[int] = []
    l0_refs: list[float] = []
    yaw_refs: list[float] = []
    left_heights: list[float] = []
    right_heights: list[float] = []
    terrain_diffs: list[float] = []
    support_rolls: list[float] = []
    leg_diff_refs: list[float] = []
    external_forces: list[float] = []
    final_reason = "not_done"

    for step in range(args.steps):
        t = step * env.p.dt
        phase = phase_for_time(t)
        ref = reference_for_phase(phase, t, args.x_limit)
        env.set_external_force_x(external_push_for_time(t, args.push_force))
        action = controller.act(state, ref)
        left_h, right_h, terrain_diff, support_roll = terrain_trace_for_state(env, state)
        next_state, reward, done, info = env.step(action, ref)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        phase_ids.append(PHASES.index(phase))
        l0_refs.append(ref.L0_ref)
        yaw_refs.append(ref.yaw_ref)
        left_heights.append(left_h)
        right_heights.append(right_h)
        terrain_diffs.append(terrain_diff)
        support_rolls.append(support_roll)
        leg_diff_refs.append(controller.last_leg_diff_ref)
        external_forces.append(env.external_force_x)

        state = next_state
        final_reason = info["final_reason"]
        if done:
            break

    out = Path(args.out) if args.out else PANORAMA_DIR / "results" / "panorama_showcase_adaptive_pd.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    states_arr = np.asarray(states)
    world_x, world_y = integrate_world_path(states_arr, env.p.dt)
    times = np.arange(len(states_arr), dtype=float) * env.p.dt

    def world_point_at(time_s: float) -> np.ndarray:
        idx = int(np.clip(np.searchsorted(times, time_s), 0, len(states_arr) - 1))
        return np.array([world_x[idx], world_y[idx]], dtype=float)

    column_xy = world_point_at(10.2)
    table_xy = world_point_at(12.9)
    support_arr = np.asarray(support_rolls)
    terrain_arr = np.asarray(terrain_diffs)
    metrics = {
        "steps": len(states_arr),
        "final_reason": final_reason,
        "success": final_reason in {"not_done", "max_steps"} and np.max(np.abs(states_arr[:, 0])) < 0.8 and np.max(np.abs(states_arr[:, 4])) < 0.8 and np.max(np.abs(states_arr[:, 8])) < 0.35,
        "max_abs_theta_rad": float(np.max(np.abs(states_arr[:, 0]))),
        "max_abs_phi_rad": float(np.max(np.abs(states_arr[:, 4]))),
        "max_abs_roll_deg": float(np.rad2deg(np.max(np.abs(states_arr[:, 8])))),
        "max_abs_support_roll_deg": float(np.rad2deg(np.max(np.abs(support_arr)))),
        "max_abs_terrain_diff_m": float(np.max(np.abs(terrain_arr))),
        "max_abs_leg_diff_m": float(np.max(np.abs(states_arr[:, 12]))),
        "max_abs_x_m": float(np.max(np.abs(world_x))),
        "max_abs_y_m": float(np.max(np.abs(world_y))),
    }
    np.savez(
        out,
        states=states_arr,
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        phase_ids=np.asarray(phase_ids),
        phase_names=np.asarray(PHASES),
        l0_refs=np.asarray(l0_refs),
        yaw_refs=np.asarray(yaw_refs),
        left_heights=np.asarray(left_heights),
        right_heights=np.asarray(right_heights),
        terrain_diffs=terrain_arr,
        support_rolls=support_arr,
        leg_diff_refs=np.asarray(leg_diff_refs),
        external_forces=np.asarray(external_forces),
        world_x=world_x,
        world_y=world_y,
        obstacle_center=np.array([1.55, 0.0, 0.22]),
        obstacle_size=np.array([0.36, 0.32, 0.44]),
        height_column_center=np.array([column_xy[0], column_xy[1], 0.16]),
        height_column_size=np.array([0.16, 0.16, 0.32]),
        height_table_top_center=np.array([table_xy[0], table_xy[1], 0.4025]),
        height_table_top_size=np.array([0.46, 0.58, 0.035]),
        height_table_leg_size=np.array([0.035, 0.035, 0.385]),
        dt=np.array([env.p.dt]),
        controller=np.array(["Panorama Adaptive-PD/VMC"]),
        terrain_mode=np.array(["panorama"]),
        final_reason=np.array([final_reason]),
        **{k: np.array([v]) for k, v in metrics.items() if k != "final_reason"},
    )
    print("task=panorama_showcase")
    print("phases=balance_walk, obstacle_turn, height, adaptive_terrain")
    print(f"steps={metrics['steps']} final_reason={final_reason} success={metrics['success']}")
    print(f"max_abs_roll_deg={metrics['max_abs_roll_deg']:.3f}")
    print(f"max_abs_support_roll_deg={metrics['max_abs_support_roll_deg']:.3f}")
    print(f"max_abs_terrain_diff_m={metrics['max_abs_terrain_diff_m']:.4f}")
    print(f"max_abs_leg_diff_m={metrics['max_abs_leg_diff_m']:.4f}")
    print(f"saved={out.resolve()}")


if __name__ == "__main__":
    main()
