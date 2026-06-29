from __future__ import annotations

import pprint
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wheel_legged.bridge.action_adapter import (
    PMPCActionBounds,
    clip_pmpc_action6,
    pmpc_action6_to_isaac_vmc_action6,
    pmpc_action6_to_python_action6,
)
from wheel_legged.bridge.state_adapter import python_state14_to_reduced
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv


def assert_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"{name} contains non-finite values: {values}")


def main() -> None:
    env = WheelLeggedEnv(task="balance_turn_roll")
    ref = Reference(yaw_ref=0.3, v_ref=0.15, L0_ref=0.34)
    state = env.reset(ref=ref, L0_init=ref.L0_ref)
    u_pmpc = np.array([0.02, 0.01, 0.34, 0.02, 1.0, 0.2], dtype=float)
    bounds = PMPCActionBounds()

    reduced_state = python_state14_to_reduced(state, env=env)
    python_action6, info = pmpc_action6_to_python_action6(u_pmpc, state, ref, env, bounds=bounds)
    isaac_action6 = pmpc_action6_to_isaac_vmc_action6(u_pmpc, bounds=bounds)
    clipped = clip_pmpc_action6(u_pmpc, bounds=bounds)

    assert reduced_state.shape == (14,)
    assert python_action6.shape == (6,)
    assert isaac_action6.shape == (6,)
    assert clipped.shape == (6,)
    assert_finite("reduced_state", reduced_state)
    assert_finite("python_action6", python_action6)
    assert_finite("isaac_action6", isaac_action6)
    assert_finite("clipped_pmpc_action6", clipped)
    if not bounds.contains(clipped):
        raise AssertionError(f"clipped action out of bounds: {clipped}")

    print("reduced_state=")
    print(np.array2string(reduced_state, precision=5, suppress_small=True))
    print("python_action6=")
    print(np.array2string(python_action6, precision=5, suppress_small=True))
    print("isaac_action6=")
    print(np.array2string(isaac_action6, precision=5, suppress_small=True))
    print("info=")
    pprint.pp(info)
    print("state_adapter OK")
    print("action_to_python OK")
    print("action_to_isaac OK")
    print("bounds OK")


if __name__ == "__main__":
    main()
