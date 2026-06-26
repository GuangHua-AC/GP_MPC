from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from wheel_legged.controllers import WheelLeggedPDController
from wheel_legged.dynamics.env import Reference, WheelLeggedEnv


def test_env_pd_step_is_finite() -> None:
    env = WheelLeggedEnv(task="terrain")
    ref = Reference(yaw_ref=0.2, v_ref=0.1, L0_ref=0.34)
    state = env.reset(ref=ref)
    controller = WheelLeggedPDController(env)
    for _ in range(20):
        action = controller.act(state, ref)
        state, reward, done, info = env.step(action, ref)
        assert np.all(np.isfinite(state))
        assert np.isfinite(reward)
        if done:
            assert info["final_reason"] != "nan_or_inf"
            break

