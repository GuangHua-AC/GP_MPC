from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from wheel_legged.utils.data import collect_transitions, save_dataset
from wheel_legged.utils.paths import DATA_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="terrain", choices=["balance", "balance_turn", "balance_turn_roll", "height", "terrain"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--noise-scale", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push-probability", type=float, default=0.0)
    parser.add_argument("--push-force", type=float, default=0.0)
    parser.add_argument("--push-duration-steps", type=int, default=20)
    parser.add_argument("--T-limit", type=float, default=None)
    parser.add_argument("--Tp-limit", type=float, default=None)
    parser.add_argument("--roll-centrifugal-ff-scale", type=float, default=0.0)
    args = parser.parse_args()
    ensure_dirs()
    states, actions, next_states = collect_transitions(
        args.task,
        args.episodes,
        args.steps,
        args.noise_scale,
        args.seed,
        push_probability=args.push_probability,
        push_force=args.push_force,
        push_duration_steps=args.push_duration_steps,
        T_limit=args.T_limit,
        Tp_limit=args.Tp_limit,
        roll_centrifugal_ff_scale=args.roll_centrifugal_ff_scale,
    )
    path = DATA_DIR / f"{args.task}_transitions.npz"
    save_dataset(path, states, actions, next_states)
    print(f"saved={path}")
    print(f"transitions={len(states)} state_dim={states.shape[1]} action_dim={actions.shape[1]}")


if __name__ == "__main__":
    main()
