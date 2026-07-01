from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

from wheel_legged.utils.paths import OUTPUT_DIR


ZONE_NAMES = np.asarray(
    [
        "Balance",
        "Turn",
        "Height",
        "Known Terrain",
        "Terrain Adaptive",
        "Jump",
    ]
)


@dataclass(frozen=True)
class ZoneSpec:
    key: str
    name: str
    task_label: str
    x0: float
    x1: float
    patterns: tuple[str, ...]
    fallback_note: str


ZONES = [
    ZoneSpec(
        key="balance",
        name="Balance",
        task_label="Balance / 30N push recovery",
        x0=0.0,
        x1=5.0,
        patterns=("outputs/balance/pmpc/*Uw5*Cw20*Gw20*seed0*pushStart1*gp_pmpc.npz",),
        fallback_note="schematic balance push",
    ),
    ZoneSpec(
        key="turn",
        name="Turn",
        task_label="Turn / yaw tracking + roll suppression",
        x0=5.0,
        x1=10.0,
        patterns=("outputs/turn/pmpc/*target30deg*v0p15*Uw5*Cw20*Gw20*seed0*.npz",),
        fallback_note="schematic yaw tracking",
    ),
    ZoneSpec(
        key="height",
        name="Height",
        task_label="Height / step L0 tracking",
        x0=10.0,
        x1=15.0,
        patterns=("outputs/height/pmpc/tracking/*step*Uw5*Cw20*Gw20*seed0*.npz",),
        fallback_note="schematic height tracking",
    ),
    ZoneSpec(
        key="known_terrain",
        name="Known Terrain",
        task_label="Known terrain / leg_diff tracking",
        x0=15.0,
        x1=20.0,
        patterns=(
            "outputs/terrain/pmpc/tuning/*Cw200*Gw50*N0p03*seed0*.npz",
            "outputs/terrain/pmpc/*Uw5*Cw20*Gw20*seed0*.npz",
        ),
        fallback_note="schematic known terrain",
    ),
    ZoneSpec(
        key="terrain_adaptive",
        name="Terrain Adaptive",
        task_label="Terrain adaptive / blind feedback",
        x0=20.0,
        x1=25.0,
        patterns=("outputs/terrain_adaptive/pmpc/*Ag0p5*seed0*.npz",),
        fallback_note="schematic adaptive terrain",
    ),
    ZoneSpec(
        key="jump",
        name="Jump",
        task_label="Jump / smoke tests / exploratory",
        x0=25.0,
        x1=30.0,
        patterns=(
            "outputs/jump/npz/jump_2d_xz_pitch_smoke.npz",
            "outputs/jump/npz/jump_2d_pitch_smoke.npz",
            "outputs/jump/npz/jump_1d_smoke.npz",
        ),
        fallback_note="schematic jump smoke",
    ),
]


def scalar(data: np.lib.npyio.NpzFile, key: str, default):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if isinstance(default, str):
        return str(value)
    try:
        return type(default)(value)
    except (TypeError, ValueError):
        return value


def find_source(zone: ZoneSpec) -> Path | None:
    for pattern in zone.patterns:
        matches = sorted(Path(".").glob(pattern))
        if matches:
            return matches[-1]
    return None


def resample_indices(n_src: int, n_dst: int) -> np.ndarray:
    if n_src <= 1:
        return np.zeros(n_dst, dtype=int)
    return np.clip(np.round(np.linspace(0, n_src - 1, n_dst)).astype(int), 0, n_src - 1)


def state_column(states: np.ndarray, idx: int, default: float, n: int) -> np.ndarray:
    if states.ndim == 2 and states.shape[1] > idx:
        return states[:, idx].astype(float)
    return np.full(n, default, dtype=float)


def l0_from_state(states: np.ndarray, n: int, default: float = 0.30) -> np.ndarray:
    if states.ndim == 2 and states.shape[1] > 10:
        return np.clip(0.30 + 0.10 * np.sin(states[:, 10].astype(float)), 0.22, 0.42)
    return np.full(n, default, dtype=float)


def action_norms(data: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    if "actions" not in data.files:
        return np.zeros(n, dtype=float)
    actions = np.asarray(data["actions"], dtype=float)
    if actions.ndim != 2:
        return np.zeros(n, dtype=float)
    return np.linalg.norm(actions, axis=1)


def metric_text_for_zone(zone: ZoneSpec, data: np.lib.npyio.NpzFile | None, source_ok: bool) -> str:
    if not source_ok or data is None:
        return f"{zone.task_label}\nsource: schematic fallback"
    final_reason = scalar(data, "final_reason", "unknown")
    if zone.key == "balance":
        states = np.asarray(data["states"], dtype=float)
        return (
            f"{zone.task_label}\n"
            f"reason={final_reason}, |theta|max={np.max(np.abs(states[:, 0])):.3f} rad, |phi|max={np.max(np.abs(states[:, 4])):.3f} rad"
        )
    if zone.key == "turn":
        yaw_err = scalar(data, "final_yaw_error_deg", np.nan)
        max_roll = scalar(data, "max_abs_roll", np.nan)
        return f"{zone.task_label}\nreason={final_reason}, yaw_err={yaw_err:.2f} deg, |roll|max={max_roll:.3f} rad"
    if zone.key == "height":
        rmse = scalar(data, "rmse_L0_error_after_1s", np.nan)
        max_err = scalar(data, "max_abs_L0_error_after_1s", np.nan)
        return f"{zone.task_label}\nreason={final_reason}, L0_RMSE={rmse:.3f} m, max_err={max_err:.3f} m"
    if zone.key == "known_terrain":
        leg_err = scalar(data, "max_abs_leg_diff_error", np.nan)
        max_roll = scalar(data, "max_abs_roll", np.nan)
        return f"{zone.task_label}\nreason={final_reason}, leg_err={leg_err:.3f} m, |roll|max={max_roll:.3f} rad"
    if zone.key == "terrain_adaptive":
        max_roll = scalar(data, "max_abs_roll", np.nan)
        if not np.isfinite(max_roll) and "roll_values" in data.files:
            max_roll = float(np.max(np.abs(np.asarray(data["roll_values"], dtype=float))))
        return f"{zone.task_label}\nreason={final_reason}, |roll|max={max_roll:.3f} rad, terrain input=unknown"
    if zone.key == "jump":
        h_target = scalar(data, "h_target", np.nan)
        success = scalar(data, "success", "unknown")
        return f"{zone.task_label}\nh_target={h_target:.3f} m, success={success}"
    return f"{zone.task_label}\nreason={final_reason}"


def schematic_zone(zone: ZoneSpec, n: int) -> dict[str, np.ndarray | str]:
    tau = np.linspace(0.0, 1.0, n)
    zeros = np.zeros(n, dtype=float)
    theta = 0.03 * np.exp(-4.0 * tau) * np.cos(6.0 * tau)
    phi = 0.02 * np.exp(-3.0 * tau)
    yaw = zeros.copy()
    roll = zeros.copy()
    l0 = np.full(n, 0.30)
    leg_diff = zeros.copy()
    robot_z = np.zeros(n, dtype=float)
    event = np.zeros(n, dtype=int)
    action_norm = 0.2 + 0.05 * np.sin(2 * np.pi * tau)

    if zone.key == "balance":
        event[(tau > 0.22) & (tau < 0.36)] |= 1
    elif zone.key == "turn":
        yaw = np.deg2rad(30.0) * (1.0 - np.exp(-5.0 * tau))
        roll = 0.04 * np.sin(np.pi * tau) * np.exp(-0.5 * tau)
        event |= 2
    elif zone.key == "height":
        l0 = np.where(tau < 0.33, 0.30, np.where(tau < 0.66, 0.36, 0.28))
        event |= 4
    elif zone.key == "known_terrain":
        leg_diff = 0.04 * ((tau > 0.35) & (tau < 0.68))
        roll = 0.02 * np.sin(np.pi * tau)
        event |= 8
    elif zone.key == "terrain_adaptive":
        leg_diff = 0.025 * np.sin(4 * np.pi * tau)
        roll = 0.035 * np.sin(3 * np.pi * tau) * np.exp(-0.2 * tau)
        event |= 16
    elif zone.key == "jump":
        robot_z = 0.45 * np.sin(np.pi * tau).clip(min=0.0)
        theta = 0.04 * np.sin(2 * np.pi * tau)
        l0 = 0.28 + 0.04 * np.sin(np.pi * tau)
        event |= 32

    return {
        "theta": theta,
        "phi": phi,
        "yaw": yaw,
        "roll": roll,
        "l0": l0,
        "leg_diff": leg_diff,
        "robot_z": robot_z,
        "action_norm": action_norm,
        "event": event,
        "source_note": zone.fallback_note,
    }


def load_zone_data(zone: ZoneSpec, n: int) -> tuple[dict[str, np.ndarray | str], Path | None, str]:
    source = find_source(zone)
    if source is None:
        return schematic_zone(zone, n), None, "fallback"

    data = np.load(source, allow_pickle=True)
    if zone.key == "jump":
        src_n = len(np.asarray(data["z"])) if "z" in data.files else len(np.asarray(data["leg_length"]))
        idx = resample_indices(src_n, n)
        theta = np.asarray(data["theta"], dtype=float)[idx] if "theta" in data.files else np.zeros(n)
        robot_z = np.asarray(data["z"], dtype=float)[idx] if "z" in data.files else np.zeros(n)
        robot_z = np.maximum(0.0, robot_z - float(np.min(robot_z)))
        l0 = np.asarray(data["leg_length"], dtype=float)[idx] if "leg_length" in data.files else np.full(n, 0.30)
        action_norm = np.zeros(n, dtype=float)
        if "Fz" in data.files:
            action_norm += np.abs(np.asarray(data["Fz"], dtype=float)[idx]) / 300.0
        if "Fx" in data.files:
            action_norm += np.abs(np.asarray(data["Fx"], dtype=float)[idx]) / 80.0
        return (
            {
                "theta": theta,
                "phi": theta,
                "yaw": np.zeros(n),
                "roll": np.zeros(n),
                "l0": np.clip(l0, 0.20, 0.45),
                "leg_diff": np.zeros(n),
                "robot_z": robot_z,
                "action_norm": action_norm,
                "event": np.full(n, 32, dtype=int),
                "source_note": "real jump smoke npz",
                "metric": metric_text_for_zone(zone, data, True),
            },
            source,
            "real",
        )

    states = np.asarray(data["states"], dtype=float)
    src_n = len(states)
    idx = resample_indices(src_n, n)
    sampled = states[idx]
    action_norm = action_norms(data, src_n)[idx]
    l0 = l0_from_state(states, src_n)[idx]

    if zone.key == "height":
        if "L0_values" in data.files:
            l0 = np.asarray(data["L0_values"], dtype=float)[idx]
        elif "L0s" in data.files:
            l0 = np.asarray(data["L0s"], dtype=float)[idx]
    if zone.key in {"known_terrain", "terrain_adaptive"}:
        if "leg_diff_values" in data.files:
            leg_diff = np.asarray(data["leg_diff_values"], dtype=float)[idx]
        else:
            leg_diff = sampled[:, 12]
        if "roll_values" in data.files:
            roll = np.asarray(data["roll_values"], dtype=float)[idx]
        else:
            roll = sampled[:, 8]
    else:
        leg_diff = sampled[:, 12] if sampled.shape[1] > 12 else np.zeros(n)
        roll = sampled[:, 8] if sampled.shape[1] > 8 else np.zeros(n)

    event = np.zeros(n, dtype=int)
    if zone.key == "balance":
        if "push_forces" in data.files:
            push = np.asarray(data["push_forces"], dtype=float)[idx]
            event[push > 1e-6] |= 1
        else:
            event[int(0.22 * n) : int(0.36 * n)] |= 1
    elif zone.key == "turn":
        event |= 2
    elif zone.key == "height":
        event |= 4
    elif zone.key == "known_terrain":
        event |= 8
    elif zone.key == "terrain_adaptive":
        event |= 16

    return (
        {
            "theta": sampled[:, 0],
            "phi": sampled[:, 4] if sampled.shape[1] > 4 else sampled[:, 0],
            "yaw": sampled[:, 6] if sampled.shape[1] > 6 else np.zeros(n),
            "roll": roll,
            "l0": np.clip(l0, 0.20, 0.45),
            "leg_diff": leg_diff,
            "robot_z": np.zeros(n),
            "action_norm": action_norm,
            "event": event,
            "source_note": "real task npz",
            "metric": metric_text_for_zone(zone, data, True),
        },
        source,
        "real",
    )


def terrain_height(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    y += 0.10 * np.exp(-0.5 * ((x - 7.4) / 1.0) ** 2)
    y += np.where((x > 11.0) & (x < 12.1), 0.20, 0.0)
    y += np.where((x > 13.0) & (x < 14.0), -0.08, 0.0)
    y += np.where((x > 16.2) & (x < 17.7), 0.18, 0.0)
    rough = (x > 20.0) & (x < 25.0)
    y += rough * (0.08 * np.sin((x - 20.0) * 4.2) + 0.035 * np.sin((x - 20.0) * 10.5))
    gap = (x > 26.25) & (x < 27.45)
    y += np.where(gap, -0.55, 0.0)
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone-duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default=str(OUTPUT_DIR / "panorama" / "capability_scene" / "capability_scene_v2.npz"))
    args = parser.parse_args()

    frames_per_zone = int(round(args.zone_duration * args.fps))
    total_frames = frames_per_zone * len(ZONES)
    time = np.arange(total_frames, dtype=float) / float(args.fps)

    zone_id = np.zeros(total_frames, dtype=int)
    zone_name = np.empty(total_frames, dtype="<U32")
    scene_x = np.zeros(total_frames, dtype=float)
    scene_y = np.zeros(total_frames, dtype=float)
    robot_theta = np.zeros(total_frames, dtype=float)
    robot_phi = np.zeros(total_frames, dtype=float)
    robot_yaw = np.zeros(total_frames, dtype=float)
    robot_roll = np.zeros(total_frames, dtype=float)
    robot_l0 = np.zeros(total_frames, dtype=float)
    robot_leg_diff = np.zeros(total_frames, dtype=float)
    robot_z = np.zeros(total_frames, dtype=float)
    action_norm = np.zeros(total_frames, dtype=float)
    metric_texts = np.empty(total_frames, dtype="<U220")
    event_flags = np.zeros(total_frames, dtype=int)
    source_task = np.empty(total_frames, dtype="<U32")
    source_file = np.empty(total_frames, dtype="<U260")
    source_status = np.empty(total_frames, dtype="<U16")
    source_summary: list[tuple[str, str, str]] = []

    for zid, zone in enumerate(ZONES):
        start = zid * frames_per_zone
        stop = start + frames_per_zone
        n = stop - start
        tau = np.linspace(0.0, 1.0, n)
        loaded, source, status = load_zone_data(zone, n)
        x = zone.x0 + (zone.x1 - zone.x0) * tau
        y = 0.55 * np.sin(np.clip((x - 5.0) / 5.0, 0.0, 1.0) * np.pi) if zone.key == "turn" else np.zeros(n)
        if zone.key == "terrain_adaptive":
            y = 0.10 * np.sin(2.0 * np.pi * tau)
        if zone.key == "jump":
            y = np.zeros(n)

        metric = str(loaded.get("metric", metric_text_for_zone(zone, None, False)))
        src = str(source.resolve()) if source is not None else "missing"
        source_summary.append((zone.name, status, src))

        zone_id[start:stop] = zid
        zone_name[start:stop] = zone.name
        scene_x[start:stop] = x
        scene_y[start:stop] = y
        robot_theta[start:stop] = np.asarray(loaded["theta"], dtype=float)
        robot_phi[start:stop] = np.asarray(loaded["phi"], dtype=float)
        robot_yaw[start:stop] = np.asarray(loaded["yaw"], dtype=float)
        robot_roll[start:stop] = np.asarray(loaded["roll"], dtype=float)
        robot_l0[start:stop] = np.asarray(loaded["l0"], dtype=float)
        robot_leg_diff[start:stop] = np.asarray(loaded["leg_diff"], dtype=float)
        robot_z[start:stop] = np.asarray(loaded["robot_z"], dtype=float)
        action_norm[start:stop] = np.asarray(loaded["action_norm"], dtype=float)
        metric_texts[start:stop] = metric
        event_flags[start:stop] = np.asarray(loaded["event"], dtype=int)
        source_task[start:stop] = zone.key
        source_file[start:stop] = src
        source_status[start:stop] = status

    terrain_x = np.linspace(-0.5, 30.5, 1600)
    terrain_profile = np.column_stack([terrain_x, terrain_height(terrain_x)])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        time=time,
        fps=np.array([args.fps]),
        zone_duration=np.array([args.zone_duration]),
        zone_id=zone_id,
        zone_name=zone_name,
        zone_names=ZONE_NAMES,
        scene_x=scene_x,
        scene_y=scene_y,
        robot_theta=robot_theta,
        robot_phi=robot_phi,
        robot_yaw=robot_yaw,
        robot_roll=robot_roll,
        robot_L0=robot_l0,
        robot_leg_diff=robot_leg_diff,
        robot_z=robot_z,
        action_norm=action_norm,
        metric_texts=metric_texts,
        terrain_profile=terrain_profile,
        event_flags=event_flags,
        source_task=source_task,
        source_file=source_file,
        source_status=source_status,
        route=np.asarray(["Balance -> Turn -> Height -> Known Terrain -> Adaptive Terrain -> Jump"]),
    )

    print("task=capability_scene_v2")
    print(f"saved={out.resolve()}")
    print(f"frames={total_frames} fps={args.fps} duration={time[-1] + 1.0 / args.fps:.2f}s")
    for name, status, src in source_summary:
        print(f"zone={name} status={status} source={src}")


if __name__ == "__main__":
    main()
