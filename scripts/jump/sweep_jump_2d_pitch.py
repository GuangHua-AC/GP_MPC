from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

JUMP_DIR = Path(__file__).resolve().parent
if str(JUMP_DIR) not in sys.path:
    sys.path.insert(0, str(JUMP_DIR))

from jump_2d_pitch_model import Jump2DPitchInitialState, Jump2DPitchModel
from jump_params import JumpParams
from test_jump_2d_pitch import compute_pitch_metrics, save_result


FIELDNAMES = [
    "h_target",
    "theta0_deg",
    "theta_dot0_deg_s",
    "max_height",
    "height_error_ratio",
    "takeoff_time",
    "landing_time",
    "flight_time",
    "max_abs_theta_deg",
    "landing_abs_theta_deg",
    "recover_abs_theta_deg",
    "max_abs_theta_dot_deg_s",
    "peak_force",
    "peak_pitch_torque",
    "pitch_torque_saturation_ratio",
    "success",
    "fail_reason",
    "npz_file",
]


def parse_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-target", type=float, default=0.03)
    parser.add_argument("--theta0-deg-list", default="0,3,-3,5,-5")
    parser.add_argument("--theta-dot0-deg-s-list", default="0")
    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--npz-dir", default="outputs/jump/npz/pitch_sweep")
    parser.add_argument("--csv", default="outputs/jump/reports/jump_2d_pitch_sweep.csv")
    args = parser.parse_args()

    rows: list[dict] = []
    npz_dir = Path(args.npz_dir)
    for theta0_deg in parse_list(args.theta0_deg_list):
        for theta_dot0_deg_s in parse_list(args.theta_dot0_deg_s_list):
            params = JumpParams(h_target=args.h_target, t_final=args.t_final)
            initial = Jump2DPitchInitialState(
                theta=float(np.deg2rad(theta0_deg)),
                theta_dot=float(np.deg2rad(theta_dot0_deg_s)),
            )
            result = Jump2DPitchModel(params, initial).run()
            metrics = compute_pitch_metrics(result, params, theta0_deg, theta_dot0_deg_s)
            tag_theta = f"{theta0_deg:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
            tag_dtheta = f"{theta_dot0_deg_s:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
            npz_file = npz_dir / f"jump_2d_pitch_h{args.h_target:.3f}_theta{tag_theta}_dtheta{tag_dtheta}.npz"
            save_result(npz_file, result, params, theta0_deg, theta_dot0_deg_s, metrics)
            row = dict(metrics)
            row["npz_file"] = str(npz_file.resolve())
            rows.append(row)
            print(
                "theta0={theta0_deg:+.1f}deg dtheta0={theta_dot0_deg_s:+.1f}deg/s "
                "max_theta={max_abs_theta_deg:.3f}deg landing_theta={landing_abs_theta_deg:.3f}deg "
                "recover_theta={recover_abs_theta_deg:.3f}deg torque_sat={pitch_torque_saturation_ratio:.3f} "
                "success={success} reason={fail_reason}".format(**row)
            )

    csv_path = Path(args.csv)
    write_csv(rows, csv_path)
    print(f"saved_csv={csv_path.resolve()}")


if __name__ == "__main__":
    main()
