from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from wheel_legged.models import NNDynamicsModel
from wheel_legged.utils.data import load_dataset
from wheel_legged.utils.paths import DATA_DIR, ensure_dirs, task_output_subdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="terrain")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--backend", default="sklearn", choices=["sklearn", "torch"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    states, actions, next_states = load_dataset(DATA_DIR / f"{args.task}_transitions.npz")
    print(f"dataset={DATA_DIR / f'{args.task}_transitions.npz'}")
    print(f"samples={len(states)} state_dim={states.shape[1]} action_dim={actions.shape[1]}")
    print(f"training backend={args.backend} epochs={args.epochs} hidden={args.hidden}")
    if args.backend == "torch":
        from wheel_legged.models.torch_dynamics import TorchDynamicsModel

        model = TorchDynamicsModel(
            states.shape[1],
            actions.shape[1],
            hidden=args.hidden,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"torch_device={model.device}")
        losses = model.fit(states, actions, next_states, epochs=args.epochs, verbose=args.verbose)
        out = task_output_subdir(args.task, "models") / f"{args.task}_nn_torch.pt"
    else:
        model = NNDynamicsModel(states.shape[1], actions.shape[1], hidden=args.hidden, device=args.device)
        model.model.verbose = args.verbose
        losses = model.fit(states, actions, next_states, epochs=args.epochs)
        out = task_output_subdir(args.task, "models") / f"{args.task}_nn.pt"
    model.save(out)
    print(f"saved={out}")
    print(f"loss_start={losses[0]:.6f}")
    print(f"final_loss={losses[-1]:.6f}")


if __name__ == "__main__":
    main()
