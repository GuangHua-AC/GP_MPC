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
    parser.add_argument("--input", default="outputs/jump/npz/jump_1d_smoke.npz")
    parser.add_argument("--out", default="outputs/jump/figures/jump_1d_smoke.png")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    t = np.asarray(data["t"], dtype=float)
    phase = np.asarray(data["phase"], dtype=int)
    phase_table = [str(x) for x in np.asarray(data["phase_name_table"]).reshape(-1)]

    plt = _require_matplotlib()
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(t, data["z_b"], label="z_b body")
    axes[0].plot(t, data["z_w"], label="z_w wheel")
    axes[0].plot(t, data["z_com"], "--", label="z_com")
    axes[0].set_ylabel("height / m")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t, data["L"], label="L")
    axes[1].plot(t, data["L_ref"], "--", label="L_ref")
    axes[1].set_ylabel("leg length / m")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(t, data["F_leg"], label="F_leg")
    axes[2].plot(t, data["F_raw"], "--", alpha=0.6, label="F_raw")
    axes[2].set_ylabel("force / N")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].step(t, phase, where="post", label="phase")
    axes[3].set_yticks(np.arange(len(phase_table)))
    axes[3].set_yticklabels(phase_table)
    axes[3].set_xlabel("time / s")
    axes[3].set_ylabel("phase")
    axes[3].grid(True, alpha=0.3)

    title = (
        f"1D Jump Smoke | target={float(scalar(data, 'h_target', 0.0)):.3f} m, "
        f"max_height={float(scalar(data, 'max_height', 0.0)):.3f} m, "
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
