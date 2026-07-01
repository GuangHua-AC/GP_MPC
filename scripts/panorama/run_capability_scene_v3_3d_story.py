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
from run_capability_scene_v3_3d import moving_average  # noqa: E402


def smooth_heading(scene_x: np.ndarray, scene_y: np.ndarray, fps: int) -> np.ndarray:
    dx = np.gradient(scene_x)
    dy = np.gradient(scene_y)
    heading = np.unwrap(np.arctan2(dy, np.maximum(dx, 1e-9)))
    sin_s = moving_average(np.sin(heading), max(5, int(0.65 * fps)))
    cos_s = moving_average(np.cos(heading), max(5, int(0.65 * fps)))
    return np.arctan2(sin_s, cos_s)


def object_bboxes(known_fix: bool = False) -> np.ndarray:
    # name_id, x_min, x_max, y_min, y_max, z_min, z_max
    known_y_min = 0.2175 if known_fix else 0.32
    known_y_max = 0.2925 if known_fix else 0.68
    known_z_max = 0.09 if known_fix else 0.18
    return np.asarray(
        [
            [1, 6.95, 7.55, -0.24, 0.24, 0.00, 0.46],  # turn road block
            [2, 11.25, 11.85, 0.28, 0.62, 0.00, 0.22],  # height-up ground obstacle
            [3, 13.05, 14.10, -0.38, 0.38, 0.62, 0.70],  # height-down overhead beam
            [4, 16.375, 17.725, known_y_min, known_y_max, 0.00, known_z_max],  # known terrain side marker
            [5, 26.25, 27.45, -0.56, 0.56, -0.55, 0.00],  # jump gap
        ],
        dtype=float,
    )


def layout_terrain_height(x: np.ndarray, include_known_step: bool = True) -> np.ndarray:
    z = np.zeros_like(x, dtype=float)
    if include_known_step:
        known = (x >= 16.25) & (x <= 18.05)
        z += np.where(known, 0.16, 0.0)
    rough = (x >= 20.0) & (x <= 25.0)
    z += rough * (0.08 * np.sin((x - 20.0) * 4.2) + 0.035 * np.sin((x - 20.0) * 10.5))
    gap = (x > 26.25) & (x < 27.45)
    z += np.where(gap, -0.55, 0.0)
    return z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone-duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--variant", choices=["story", "layout", "motion", "motion_known_fix"], default="story")
    parser.add_argument("--heading-warning-deg", type=float, default=45.0)
    parser.add_argument("--out", default=str(OUTPUT_DIR / "panorama" / "capability_scene" / "capability_scene_v3_3d_story.npz"))
    args = parser.parse_args()
    if args.variant == "layout" and args.out.endswith("capability_scene_v3_3d_story.npz"):
        args.out = str(OUTPUT_DIR / "panorama" / "capability_scene" / "capability_scene_v3_3d_story_layout.npz")
    if args.variant == "motion" and args.out.endswith("capability_scene_v3_3d_story.npz"):
        args.out = str(OUTPUT_DIR / "panorama" / "capability_scene" / "capability_scene_v3_3d_story_motion.npz")
    if args.variant == "motion_known_fix" and args.out.endswith("capability_scene_v3_3d_story.npz"):
        args.out = str(OUTPUT_DIR / "panorama" / "capability_scene" / "capability_scene_v3_3d_story_motion_known_fix.npz")
    motion_variant = args.variant.startswith("motion")

    frames_per_zone = int(round(args.zone_duration * args.fps))
    total_frames = frames_per_zone * len(ZONES)
    time = np.arange(total_frames, dtype=float) / float(args.fps)

    zone_id = np.zeros(total_frames, dtype=int)
    zone_name = np.empty(total_frames, dtype="<U32")
    scene_x = np.zeros(total_frames, dtype=float)
    scene_y = np.zeros(total_frames, dtype=float)
    robot_theta = np.zeros(total_frames, dtype=float)
    robot_phi = np.zeros(total_frames, dtype=float)
    robot_roll = np.zeros(total_frames, dtype=float)
    robot_l0 = np.zeros(total_frames, dtype=float)
    robot_leg_diff = np.zeros(total_frames, dtype=float)
    robot_z = np.zeros(total_frames, dtype=float)
    action_norm = np.zeros(total_frames, dtype=float)
    metric_texts = np.empty(total_frames, dtype="<U260")
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
            # Smooth obstacle bypass: starts/ends on centerline with zero slope.
            y = 0.82 * np.sin(np.pi * tau) ** 2
        elif zone.key == "terrain_adaptive":
            y = 0.14 * np.sin(2.0 * np.pi * tau)

        metric = str(loaded.get("metric", zone.task_label))
        if zone.key == "turn":
            metric += "\nobstacle bypass, heading smoothed"
            event_flags[start:stop] |= 64
        if zone.key == "height":
            if motion_variant:
                metric += "\nobvious L0 up/down motion, small curb, low gate"
            else:
                metric += "\nraise over block, lower under beam"
            event_flags[start:stop] |= 128
        if zone.key == "known_terrain" and motion_variant:
            metric += "\none wheel on narrow side rail, known dH, leg_diff tracking"

        src = str(source.resolve()) if source is not None else "missing"
        source_summary.append((zone.name, status, src))

        zone_id[start:stop] = zid
        zone_name[start:stop] = zone.name
        scene_x[start:stop] = x
        scene_y[start:stop] = y
        robot_theta[start:stop] = np.asarray(loaded["theta"], dtype=float)
        robot_phi[start:stop] = np.asarray(loaded["phi"], dtype=float)
        robot_roll[start:stop] = np.asarray(loaded["roll"], dtype=float)
        l0_values = np.asarray(loaded["l0"], dtype=float)
        leg_diff_values = np.asarray(loaded["leg_diff"], dtype=float)
        if motion_variant and zone.key == "height":
            raise_profile = np.exp(-0.5 * ((tau - 0.34) / 0.13) ** 2)
            lower_profile = np.exp(-0.5 * ((tau - 0.70) / 0.14) ** 2)
            l0_values = np.clip(0.30 + 0.11 * raise_profile - 0.075 * lower_profile, 0.22, 0.42)
        if motion_variant and zone.key == "known_terrain":
            support = ((tau > 0.22) & (tau < 0.78)).astype(float)
            support = moving_average(support, max(3, int(0.25 * args.fps)))
            leg_diff_values = (0.075 if args.variant == "motion_known_fix" else 0.085) * support
        robot_l0[start:stop] = l0_values
        robot_leg_diff[start:stop] = leg_diff_values
        robot_z[start:stop] = np.asarray(loaded["robot_z"], dtype=float)
        action_norm[start:stop] = np.asarray(loaded["action_norm"], dtype=float)
        metric_texts[start:stop] = metric
        event_flags[start:stop] |= np.asarray(loaded["event"], dtype=int)
        source_task[start:stop] = zone.key
        source_file[start:stop] = src
        source_status[start:stop] = status
        zone_transition_flags[start] = 1
        zone_transition_flags[stop - 1] = 1

    robot_yaw = smooth_heading(scene_x, scene_y, args.fps)
    heading_jump = np.rad2deg(np.max(np.abs(np.diff(np.unwrap(robot_yaw)))))
    if heading_jump > args.heading_warning_deg:
        print(f"warning_heading_jump_deg={heading_jump:.2f}")

    terrain_x = np.linspace(-0.5, 30.5, 1800)
    if motion_variant:
        terrain_z = layout_terrain_height(terrain_x, include_known_step=False)
    elif args.variant == "layout":
        terrain_z = layout_terrain_height(terrain_x, include_known_step=True)
    else:
        terrain_z = terrain_height(terrain_x)
    terrain_profile = np.column_stack([terrain_x, terrain_z])
    ground_profile = terrain_profile.copy()
    robot_base_z = np.interp(scene_x, terrain_x, terrain_z) + robot_z
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
        object_bboxes=object_bboxes(known_fix=args.variant == "motion_known_fix"),
        bbox_names=np.asarray(["turn_obstacle", "height_up_block", "height_down_beam", "known_marker", "jump_gap"]),
        route=np.asarray(["Balance -> Turn -> Height -> Known Terrain -> Adaptive Terrain -> Jump"]),
        scene_variant=np.asarray(
            [
                "story_motion_known_terrain_fix"
                if args.variant == "motion_known_fix"
                else "story_motion_fix"
                if args.variant == "motion"
                else "story_layout_fix"
                if args.variant == "layout"
                else "story_and_collision_fix"
            ]
        ),
        renderer_hint=np.asarray(["3D orthographic panorama with five-link robot, story objects, smooth heading"]),
    )

    print(f"task=capability_scene_v3_3d_{args.variant}")
    print(f"saved={out.resolve()}")
    print(f"frames={total_frames} fps={args.fps} duration={time[-1] + 1.0 / args.fps:.2f}s")
    print(f"max_heading_step_deg={heading_jump:.3f}")
    for name, status, src in source_summary:
        print(f"zone={name} status={status} source={src}")


if __name__ == "__main__":
    main()
