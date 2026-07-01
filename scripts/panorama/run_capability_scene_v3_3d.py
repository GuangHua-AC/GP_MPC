from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _bootstrap  # noqa: F401

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from wheel_legged.utils.paths import OUTPUT_DIR

from run_capability_scene_v2 import ZONES, ZONE_NAMES, load_zone_data, terrain_height  # noqa: E402


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone-duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default=str(OUTPUT_DIR / "panorama" / "capability_scene" / "capability_scene_v3_3d.npz"))
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
    metric_texts = np.empty(total_frames, dtype="<U240")
    event_flags = np.zeros(total_frames, dtype=int)
    source_task = np.empty(total_frames, dtype="<U32")
    source_file = np.empty(total_frames, dtype="<U280")
    source_status = np.empty(total_frames, dtype="<U16")
    zone_transition_flags = np.zeros(total_frames, dtype=int)
    source_summary: list[tuple[str, str, str]] = []

    for zid, zone in enumerate(ZONES):
        start = zid * frames_per_zone
        stop = start + frames_per_zone
        n = stop - start
        tau = np.linspace(0.0, 1.0, n)
        loaded, source, status = load_zone_data(zone, n)
        x = zone.x0 + (zone.x1 - zone.x0) * tau
        y = np.zeros(n, dtype=float)
        if zone.key == "turn":
            y = 0.58 * np.sin(np.clip((x - zone.x0) / (zone.x1 - zone.x0), 0.0, 1.0) * np.pi)
        elif zone.key == "terrain_adaptive":
            y = 0.14 * np.sin(2.0 * np.pi * tau)

        metric = str(loaded.get("metric", zone.task_label))
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
        zone_transition_flags[start] = 1
        zone_transition_flags[stop - 1] = 1

    terrain_x = np.linspace(-0.5, 30.5, 1800)
    terrain_z = terrain_height(terrain_x)
    terrain_profile = np.column_stack([terrain_x, terrain_z])
    ground_profile = terrain_profile.copy()
    robot_base_z = terrain_height(scene_x) + robot_z
    scene_pose = np.column_stack([scene_x, scene_y, robot_base_z, robot_yaw])
    robot_pose_3d_like = np.column_stack(
        [scene_x, scene_y, robot_base_z, robot_theta, robot_phi, robot_yaw, robot_roll, robot_l0, robot_leg_diff]
    )
    camera_x = moving_average(scene_x, max(3, int(0.55 * args.fps)))

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
        ground_profile=ground_profile,
        event_flags=event_flags,
        source_task=source_task,
        source_file=source_file,
        source_status=source_status,
        scene_pose=scene_pose,
        robot_pose_3d_like=robot_pose_3d_like,
        camera_x=camera_x,
        zone_transition_flags=zone_transition_flags,
        route=np.asarray(["Balance -> Turn -> Height -> Known Terrain -> Adaptive Terrain -> Jump"]),
        renderer_hint=np.asarray(["3D orthographic panorama with five-link wheel-legged robot"]),
    )

    print("task=capability_scene_v3_3d")
    print(f"saved={out.resolve()}")
    print(f"frames={total_frames} fps={args.fps} duration={time[-1] + 1.0 / args.fps:.2f}s")
    for name, status, src in source_summary:
        print(f"zone={name} status={status} source={src}")


if __name__ == "__main__":
    main()
