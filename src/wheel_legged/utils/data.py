from __future__ import annotations

from pathlib import Path

import numpy as np

from wheel_legged.controllers.pd import PDGains, WheelLeggedPDController
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv


def collect_transitions(
    task: str,
    episodes: int,
    steps: int,
    noise_scale: float,
    seed: int,
    push_probability: float = 0.0,
    push_force: float = 0.0,
    push_duration_steps: int = 20,
    T_limit: float | None = None,
    Tp_limit: float | None = None,
    roll_centrifugal_ff_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    env = WheelLeggedEnv(task=task)
    pd = WheelLeggedPDController(env, PDGains(roll_centrifugal_ff_scale=roll_centrifugal_ff_scale))
    states = []
    actions = []
    next_states = []

    for _ in range(episodes):
        ref = Reference(
            yaw_ref=float(rng.uniform(-0.35, 0.35)),
            v_ref=float(rng.uniform(-0.15, 0.25)),
            L0_ref=float(rng.uniform(0.27, 0.37)),
        )
        s = env.reset(
            theta0=float(rng.uniform(-0.06, 0.06)),
            phi0=float(rng.uniform(-0.06, 0.06)),
            roll0=float(rng.uniform(-0.04, 0.04)),
            ref=ref,
        )
        if T_limit is not None:
            env.p.T_limit = float(T_limit)
        if Tp_limit is not None:
            env.p.Tp_limit = float(Tp_limit)
        push_steps_left = 0
        current_push = 0.0
        for _step in range(steps):
            if push_steps_left <= 0 and push_probability > 0.0 and rng.random() < push_probability:
                push_steps_left = max(1, int(push_duration_steps))
                current_push = float(rng.choice([-1.0, 1.0]) * push_force)
            elif push_steps_left <= 0:
                current_push = 0.0

            env.set_external_force_x(current_push)
            a = pd.act(s, ref)
            span = np.array([env.p.T_limit, env.p.Tp_limit, env.p.Tyaw_limit, env.p.Froll_limit, env.p.Fheight_max, env.p.leg_diff_cmd_limit])
            a = a + rng.normal(0.0, noise_scale * span)
            ns, _reward, done, _info = env.step(a, ref)
            states.append(s)
            actions.append(env._clip_action(a))
            next_states.append(ns)
            s = ns
            push_steps_left -= 1
            if done:
                break

    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32), np.asarray(next_states, dtype=np.float32)


def save_dataset(path: str | Path, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, states=states, actions=actions, next_states=next_states)


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["states"], data["actions"], data["next_states"]
