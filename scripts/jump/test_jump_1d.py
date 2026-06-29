from __future__ import annotations

import argparse
import sys
from pathlib import Path

JUMP_DIR = Path(__file__).resolve().parent
if str(JUMP_DIR) not in sys.path:
    sys.path.insert(0, str(JUMP_DIR))

import numpy as np

from jump_1d_model import Jump1DModel
from jump_params import JumpParams


def save_result(out_path: Path, result: dict, params: JumpParams) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload.update(params.to_npz_payload())
    np.savez(out_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-target", type=float, default=0.03)
    parser.add_argument("--out", default="outputs/jump/npz/jump_1d_smoke.npz")
    parser.add_argument("--t-final", type=float, default=3.0)
    args = parser.parse_args()

    params = JumpParams(h_target=args.h_target, t_final=args.t_final)
    model = Jump1DModel(params)
    result = model.run()
    out = Path(args.out)
    save_result(out, result, params)

    max_height = float(result["max_height"][0])
    takeoff_time = float(result["takeoff_time"][0])
    landing_time = float(result["landing_time"][0])
    peak_force = float(result["peak_force"][0])
    success = bool(result["success"][0])
    saturation_ratio = float(np.mean(result["force_saturated"]))

    print("task=jump_1d_smoke")
    print(f"h_target={args.h_target:.4f} m")
    print(f"max_height={max_height:.5f} m")
    print(f"takeoff_time={takeoff_time:.4f} s")
    print(f"landing_time={landing_time:.4f} s")
    print(f"peak_force={peak_force:.3f} N")
    print(f"saturation_ratio={saturation_ratio:.4f}")
    print(f"success={success}")
    print(f"saved={out.resolve()}")


if __name__ == "__main__":
    main()
