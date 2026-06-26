from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from wheel_legged.models import GPDynamicsModel
from wheel_legged.utils.data import load_dataset
from wheel_legged.utils.paths import DATA_DIR, ensure_dirs, task_output_subdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="terrain")
    parser.add_argument("--max-points", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--active-dims", default=None, help="Comma separated state dims to train. Default uses task-specific dims.")
    args = parser.parse_args()
    ensure_dirs()
    states, actions, next_states = load_dataset(DATA_DIR / f"{args.task}_transitions.npz")
    if args.active_dims:
        active_dims = [int(x) for x in args.active_dims.split(",") if x.strip()]
    elif args.task == "balance":
        active_dims = [0, 1, 2, 3, 4, 5]
    elif args.task == "balance_turn":
        active_dims = [0, 1, 2, 3, 4, 5, 6, 7]
    elif args.task == "balance_turn_roll":
        active_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif args.task == "height":
        active_dims = [0, 1, 2, 3, 4, 5, 10, 11]
    else:
        active_dims = list(range(states.shape[1]))
    print(f"dataset={DATA_DIR / f'{args.task}_transitions.npz'}")
    print(f"samples={len(states)} max_points={args.max_points}")
    print(f"active_dims={active_dims}")
    model = GPDynamicsModel(states.shape[1], actions.shape[1], active_dims=active_dims)
    model.fit(states, actions, next_states, max_points=args.max_points, seed=args.seed)
    out = task_output_subdir(args.task, "models") / f"{args.task}_gp.joblib"
    model.save(out)
    print(f"saved={out}")
    print(f"trained_dims={sorted(model.models.keys())}")


if __name__ == "__main__":
    main()
