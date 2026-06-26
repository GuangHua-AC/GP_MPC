from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run(["scripts/common/01_run_pd.py", "--task", "balance_turn_roll", "--steps", "120"])
    run(["scripts/common/02_collect_data.py", "--task", "terrain", "--episodes", "2", "--steps", "80", "--noise-scale", "0.03"])
    run(["scripts/common/03_train_nn.py", "--task", "terrain", "--epochs", "3", "--hidden", "64"])
    run(["scripts/common/04_run_nn_mpc.py", "--task", "terrain", "--steps", "5", "--horizon", "3", "--candidates", "8"])
    print("smoke ok")


if __name__ == "__main__":
    main()
