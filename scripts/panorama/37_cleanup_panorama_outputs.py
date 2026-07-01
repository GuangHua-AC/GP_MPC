from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANORAMA = ROOT / "outputs" / "panorama"
ARCHIVE = PANORAMA / "archive" / "deprecated"
MANIFEST = PANORAMA / "archive" / "deprecated_manifest.csv"

KEEP_RELATIVE = {
    Path("capability_scene/capability_scene_final.npz"),
    Path("videos/capability_scene_final.mp4"),
    Path("videos/capability_scene_final.gif"),
    Path("figures/capability_scene_final_snapshot.png"),
    Path("figures/wheel_legged_task_big_map_v3_thesis.png"),
    Path("figures/wheel_legged_task_big_map_v3_thesis.pdf"),
    Path("figures/wheel_legged_task_big_map_v3_defense.png"),
    Path("figures/wheel_legged_task_big_map_v3_defense.pdf"),
    Path("figures/wheel_legged_task_big_map_v3_slide.png"),
}

DEPRECATED_PREFIXES = (
    "capability_scene_v2",
    "capability_scene_v3_3d",
    "capability_scene_v3_3d_story",
)

DEPRECATED_MAP_PREFIXES = (
    "wheel_legged_task_big_map_v2",
)


def is_candidate(path: Path) -> tuple[bool, str]:
    try:
        rel = path.relative_to(PANORAMA)
    except ValueError:
        return False, "outside_panorama"
    if rel in KEEP_RELATIVE:
        return False, "keep_final_or_documented"
    if rel.parts and rel.parts[0] == "archive":
        return False, "already_archive"
    if rel.parts and rel.parts[0] not in {"capability_scene", "videos", "figures"}:
        return False, "not_panorama_display_output"
    name = path.name
    if name.startswith(DEPRECATED_PREFIXES):
        return True, "deprecated_capability_scene_version"
    if name.startswith(DEPRECATED_MAP_PREFIXES):
        return True, "deprecated_capability_map_version"
    if "motion_check" in name or name.endswith("_check.png"):
        return True, "temporary_check_frame"
    return False, "not_deprecated"


def collect_candidates() -> list[tuple[Path, Path, str]]:
    candidates: list[tuple[Path, Path, str]] = []
    if not PANORAMA.exists():
        return candidates
    for path in PANORAMA.rglob("*"):
        if not path.is_file():
            continue
        ok, reason = is_candidate(path)
        if not ok:
            continue
        rel = path.relative_to(PANORAMA)
        candidates.append((path, ARCHIVE / rel, reason))
    return sorted(candidates, key=lambda item: str(item[0]).lower())


def write_manifest(candidates: list[tuple[Path, Path, str]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_path", "archived_path", "file_size", "reason", "timestamp"])
        for original, archived, reason in candidates:
            writer.writerow([str(original), str(archived), original.stat().st_size, reason, timestamp])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Move deprecated panorama outputs into the archive.")
    parser.add_argument("--delete-deprecated", action="store_true", help="Delete the deprecated archive directory. Requires --apply.")
    args = parser.parse_args()

    if args.delete_deprecated:
        if not args.apply:
            raise SystemExit("--delete-deprecated requires --apply")
        if ARCHIVE.exists():
            shutil.rmtree(ARCHIVE)
            print(f"deleted_deprecated={ARCHIVE}")
        else:
            print(f"deprecated_not_found={ARCHIVE}")
        return

    candidates = collect_candidates()
    write_manifest(candidates)
    print(f"manifest={MANIFEST}")
    print(f"deprecated_candidates={len(candidates)}")
    for original, archived, reason in candidates:
        print(f"{'MOVE' if args.apply else 'DRY'} {original} -> {archived} reason={reason}")

    if not args.apply:
        print("dry_run=True; add --apply to move files")
        return

    for original, archived, _reason in candidates:
        archived.parent.mkdir(parents=True, exist_ok=True)
        if archived.exists():
            archived.unlink()
        shutil.move(str(original), str(archived))
    print(f"moved={len(candidates)}")


if __name__ == "__main__":
    main()
