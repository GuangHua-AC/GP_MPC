from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def scalar(data, key: str, default=np.nan):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/jump/npz/jump_2d_pitch_smoke.npz")
    parser.add_argument("--out", default="outputs/jump/figures/jump_2d_pitch_smoke.png")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    t = np.asarray(data["t"], dtype=float)
    phase = np.asarray(data["phase"], dtype=int)
    phase_table = [str(x) for x in np.asarray(data["phase_name_table"]).reshape(-1)]

    plt = _require_matplotlib()
    fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(t, data["z"], label="z")
    axes[0].set_ylabel("height / m")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, data["leg_length"], label="leg_length")
    axes[1].plot(t, data["leg_length_ref"], "--", label="leg_length_ref")
    axes[1].set_ylabel("leg length / m")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].step(t, phase, where="post", label="phase")
    axes[2].plot(t, data["contact"] * (len(phase_table) - 1), "--", label="contact")
    axes[2].set_yticks(np.arange(len(phase_table)))
    axes[2].set_yticklabels(phase_table)
    axes[2].set_ylabel("phase/contact")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(t, np.rad2deg(data["theta"]), label="theta")
    axes[3].axhline(10.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[3].axhline(-10.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("theta / deg")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    axes[4].plot(t, np.rad2deg(data["theta_dot"]), label="theta_dot")
    axes[4].set_ylabel("theta_dot / deg/s")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="best")

    axes[5].plot(t, data["Fz"], color="tab:orange", label="Fz")
    axes[5].set_xlabel("time / s")
    axes[5].set_ylabel("Fz / N")
    axes[5].grid(True, alpha=0.3)
    torque_axis = axes[5].twinx()
    torque_axis.plot(t, data["T_pitch"], color="tab:blue", label="T_pitch")
    torque_axis.set_ylabel("T_pitch / Nm")
    lines, labels = axes[5].get_legend_handles_labels()
    lines2, labels2 = torque_axis.get_legend_handles_labels()
    axes[5].legend(lines + lines2, labels + labels2, loc="best")

    title = (
        f"2D Pitch Jump Smoke | h={float(scalar(data, 'h_target', 0.0)):.3f} m, "
        f"theta0={float(scalar(data, 'theta0_deg', 0.0)):.1f} deg, "
        f"success={bool(scalar(data, 'success', False))}"
    )
    fig.suptitle(title)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved={out.resolve()}")


if __name__ == "__main__":
    main()
