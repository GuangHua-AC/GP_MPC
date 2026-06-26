from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401
import numpy as np


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Rectangle

    return plt, FuncAnimation, FFMpegWriter, PillowWriter, Circle, Rectangle


def load_result(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if "states" not in data.files:
        raise KeyError(f"{path} does not contain states; keys={data.files}")
    if "actions" not in data.files:
        raise KeyError(f"{path} does not contain actions; keys={data.files}")

    states = np.asarray(data["states"], dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    if states.ndim != 2 or states.shape[1] < 6:
        raise ValueError(f"states must be (N, >=6), got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] < 2:
        raise ValueError(f"actions must be (N, >=2), got {actions.shape}")

    meta: dict[str, Any] = {}
    for key in data.files:
        if key in {"states", "actions"}:
            continue
        value = data[key]
        try:
            meta[key] = value.item() if value.shape == () else value
        except Exception:
            meta[key] = value
    return states, actions, meta


def output_fps_from_time(dt: float, stride: int, speed: float) -> int:
    frame_dt = dt * max(1, stride)
    fps = 1.0 / max(frame_dt / max(speed, 1e-6), 1e-6)
    return int(np.clip(round(fps), 5, 60))


class BalanceRenderer:
    def __init__(self, states: np.ndarray, actions: np.ndarray, meta: dict[str, Any], title: str):
        plt, _FuncAnimation, _FFMpegWriter, _PillowWriter, Circle, Rectangle = _require_matplotlib()
        self.plt = plt
        self.Circle = Circle
        self.Rectangle = Rectangle
        self.states = states
        self.actions = actions
        self.meta = meta
        self.title = title
        self.dt = float(meta.get("dt", 0.005))
        self.R = 0.08
        self.L = 0.32
        self.body_w = 0.28
        self.body_h = 0.16
        self.fig = plt.figure(figsize=(13, 7))
        self.ax_robot = self.fig.add_subplot(1, 2, 1)
        self.ax_info = self.fig.add_subplot(1, 2, 2)

    def _draw_body(self, ax, center_x: float, center_z: float, pitch: float) -> None:
        rect = self.Rectangle(
            (-self.body_w / 2.0, -self.body_h / 2.0),
            self.body_w,
            self.body_h,
            facecolor="#d9dde7",
            edgecolor="#222222",
            linewidth=1.5,
        )
        transform = (
            self.plt.matplotlib.transforms.Affine2D()
            .rotate(pitch)
            .translate(center_x, center_z)
            + ax.transData
        )
        rect.set_transform(transform)
        ax.add_patch(rect)

    def update(self, frame_idx: int):
        s = self.states[frame_idx]
        theta, theta_dot, x, x_dot, phi, phi_dot = s[:6]
        action = self.actions[min(frame_idx, len(self.actions) - 1)]
        T = action[0]
        Tp = action[1]
        push = 0.0
        if "push_forces" in self.meta:
            push_forces = self.meta["push_forces"]
            if frame_idx < len(push_forces):
                push = float(push_forces[frame_idx])

        ax = self.ax_robot
        ax.clear()
        ax.set_title(self.title)
        view = 0.75
        ax.set_xlim(x - view, x + view)
        ax.set_ylim(-0.05, 0.85)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x / m")
        ax.set_ylabel("z / m")
        ax.axhline(0.0, color="#555555", linewidth=1.2)

        wheel = self.Circle((x, self.R), self.R, facecolor="#111111", edgecolor="black")
        ax.add_patch(wheel)
        axle = np.array([x, self.R])
        hip = axle + np.array([self.L * np.sin(theta), self.L * np.cos(theta)])
        body_center = hip + np.array([0.0, 0.05])

        ax.plot([axle[0], hip[0]], [axle[1], hip[1]], color="#222222", linewidth=4)
        ax.plot(axle[0], axle[1], "o", color="#f2f2f2", markersize=5)
        ax.plot(hip[0], hip[1], "o", color="#222222", markersize=5)
        self._draw_body(ax, body_center[0], body_center[1], phi)

        if abs(push) > 1e-9:
            direction = 1.0 if push > 0 else -1.0
            ax.arrow(
                body_center[0] - 0.35 * direction,
                body_center[1] + 0.14,
                0.22 * direction,
                0.0,
                width=0.012,
                color="#d62728",
                length_includes_head=True,
            )
            ax.text(body_center[0] - 0.40 * direction, body_center[1] + 0.18, f"{push:g} N", color="#d62728")

        path_x = self.states[: frame_idx + 1, 2]
        ax.plot(path_x, np.zeros_like(path_x), color="#1f77b4", linewidth=2, alpha=0.8)

        info = self.ax_info
        info.clear()
        info.axis("off")
        info.set_title("Balance State")
        lines = [
            f"frame      {frame_idx}/{len(self.states) - 1}",
            f"time       {frame_idx * self.dt:.3f} s",
            "",
            f"theta      {theta:+.4f} rad ({np.rad2deg(theta):+.2f} deg)",
            f"theta_dot  {theta_dot:+.4f} rad/s",
            f"x          {x:+.4f} m",
            f"x_dot      {x_dot:+.4f} m/s",
            f"phi        {phi:+.4f} rad ({np.rad2deg(phi):+.2f} deg)",
            f"phi_dot    {phi_dot:+.4f} rad/s",
            "",
            f"T          {T:+.4f}",
            f"Tp         {Tp:+.4f}",
            f"push       {push:+.2f} N",
        ]
        info.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)

        inset = info.inset_axes([0.08, 0.05, 0.86, 0.32])
        t = np.arange(len(self.states)) * self.dt
        inset.plot(t, np.rad2deg(self.states[:, 0]), label="theta deg")
        inset.plot(t, np.rad2deg(self.states[:, 4]), label="phi deg")
        inset.plot(t, self.states[:, 2], label="x m")
        inset.axvline(frame_idx * self.dt, color="black", linewidth=1.0, alpha=0.6)
        inset.grid(True, alpha=0.25)
        inset.legend(fontsize=8, loc="upper left")
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="outputs/balance/pd/balance_pd.npz")
    parser.add_argument("--out", default=None, help="MP4 output path.")
    parser.add_argument("--gif", default=None, help="GIF output path.")
    parser.add_argument("--fps", type=int, default=None, help="Output FPS. Default preserves simulation time.")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed relative to simulation time.")
    parser.add_argument("--title", default="Balance Control")
    args = parser.parse_args()

    if args.out is None and args.gif is None:
        args.out = "outputs/balance/videos/balance.mp4"
        args.gif = "outputs/balance/videos/balance.gif"

    plt, FuncAnimation, FFMpegWriter, PillowWriter, _Circle, _Rectangle = _require_matplotlib()
    states, actions, meta = load_result(args.npz)
    renderer = BalanceRenderer(states, actions, meta, args.title)
    stride = max(1, args.stride)
    fps = args.fps if args.fps is not None else output_fps_from_time(renderer.dt, stride, args.speed)
    frames = list(range(0, len(states), stride))
    if frames[-1] != len(states) - 1:
        frames.append(len(states) - 1)

    animation = FuncAnimation(renderer.fig, renderer.update, frames=frames, interval=1000 / fps, blit=False)
    print(f"states={len(states)} dt={renderer.dt:g}s stride={stride} frames={len(frames)} fps={fps} video_duration≈{len(frames) / fps:.2f}s sim_duration≈{len(states) * renderer.dt:.2f}s")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(out), writer=FFMpegWriter(fps=fps))
        print(f"saved_mp4={out}")

    if args.gif:
        gif = Path(args.gif)
        gif.parent.mkdir(parents=True, exist_ok=True)
        gif_fps = min(fps, 30)
        animation.save(str(gif), writer=PillowWriter(fps=gif_fps))
        print(f"saved_gif={gif}")

    plt.close(renderer.fig)


if __name__ == "__main__":
    main()
