#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Split mixed trajectory rollout directories into ~/carla_town{i} by metadata town."
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="~/carla_dataset",
        help="Directory containing rollout subdirectories (each with metadata.json).",
    )
    p.add_argument(
        "--home_dir",
        type=str,
        default=str(Path.home()),
        help="Home directory where carla_town{i} folders are created (default: current user home).",
    )
    p.add_argument(
        "--skip_towns",
        type=str,
        default="2",
        help="Comma-separated town IDs to skip (default: 2).",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned operations without changing files.",
    )
    return p.parse_args()


def parse_skip_towns(skip_towns_str: str):
    out = set()
    for token in skip_towns_str.split(","):
        token = token.strip()
        if token:
            out.add(int(token))
    return out


def unique_destination_path(dst: Path) -> Path:
    if not dst.exists():
        return dst

    stem = dst.name
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_dup{i}"
        if not candidate.exists():
            return candidate
        i += 1


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    home_dir = Path(args.home_dir).expanduser().resolve()
    skip_towns = parse_skip_towns(args.skip_towns)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir is not a directory: {dataset_dir}")
    if not home_dir.is_dir():
        raise FileNotFoundError(f"home_dir is not a directory: {home_dir}")

    rollout_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    moved = 0
    skipped = 0
    errors = 0

    for rollout_dir in rollout_dirs:
        metadata_path = rollout_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"[SKIP] {rollout_dir.name}: missing metadata.json")
            skipped += 1
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            town = int(metadata["town"])
        except Exception as exc:
            print(f"[SKIP] {rollout_dir.name}: failed to parse metadata ({exc})")
            skipped += 1
            continue

        if town in skip_towns:
            print(f"[SKIP] {rollout_dir.name}: town={town}")
            skipped += 1
            continue

        town_dir = home_dir / f"carla_town{town}"
        dst = unique_destination_path(town_dir / rollout_dir.name)

        op = "COPY" if args.copy else "MOVE"
        print(f"[{op}] {rollout_dir} -> {dst}")

        if args.dry_run:
            moved += 1
            continue

        try:
            town_dir.mkdir(parents=True, exist_ok=True)
            if args.copy:
                shutil.copytree(str(rollout_dir), str(dst))
            else:
                shutil.move(str(rollout_dir), str(dst))
            moved += 1
        except Exception as exc:
            print(f"[ERROR] {rollout_dir.name}: {exc}")
            errors += 1

    print(
        f"\nDone. processed={len(rollout_dirs)} moved_or_planned={moved} skipped={skipped} errors={errors}"
    )


if __name__ == "__main__":
    main()
