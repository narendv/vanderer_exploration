#!/usr/bin/env python3
"""
Build train / test / test_unseen splits for the TartanGround subset.

Two kinds of held-out data are produced:

  test        - trajectories from environments that ARE also in train
                ("seen scene, unseen trajectory")
  test_unseen - every trajectory from a small number of environments that are
                held out entirely ("unseen scene")

Held-out environments must be scene *singletons*. TartanGround ships several
environments that are the same physical scene under a different season or time
of day (OldTown{Fall,Night,Summer,Winter}, SeasonalForest*, WaterMill{Day,Night},
ModularNeighborhood{,IntExt}). Holding out one member of such a family leaks the
geometry through its siblings, so the unseen split would not measure what it
claims to.

Trajectory names match the flattened layout written by the conversion step:
    <EnvName>_<P1xxx>       e.g. Gascola_P1000
"""

import argparse
import glob
import os
import random
from collections import defaultdict
from typing import Dict, List

# Environments held out entirely. Both are scene singletons (no seasonal/
# time-of-day sibling elsewhere in the tree) and are the smallest such
# environment in their category.
HELD_OUT_ENVS = {
    "SeasideTown": "urban",       # 2 trajs,  1,494 frames
    "GothicIsland": "rural",      # 3 trajs,  3,883 frames
}

# Environments with fewer than this many trajectories contribute nothing to the
# seen-scene test split -- taking one of two trajectories would remove half the
# environment from training.
MIN_TRAJS_FOR_SEEN_TEST = 4

# Fraction of all frames to place in the seen-scene test split. This is a frame
# budget rather than a per-environment quota: taking one trajectory from every
# eligible environment overshoots badly, because trajectory lengths range from
# 337 to 8,443 frames.
DEFAULT_SEEN_FRAC = 0.09


def scan(root: str) -> Dict[str, List[str]]:
    """Map environment name -> sorted list of flattened trajectory names."""
    envs: Dict[str, List[str]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "Data_diff", "P*"))):
        env = path.split(os.sep)[-3]
        envs[env].append(f"{env}_{os.path.basename(path)}")
    return dict(envs)


def frame_count(root: str, traj_name: str) -> int:
    env, traj = traj_name.rsplit("_", 1)
    pose = os.path.join(root, env, "Data_diff", traj, "pose_lcam_front.txt")
    with open(pose, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build(root: str, seen_frac: float, seed: int):
    envs = scan(root)

    missing = set(HELD_OUT_ENVS) - set(envs)
    if missing:
        raise SystemExit(f"Held-out environments not found on disk: {sorted(missing)}")

    rng = random.Random(seed)
    frames = {t: frame_count(root, t)
              for env_trajs in envs.values() for t in env_trajs}

    test_unseen: List[str] = []
    for env in sorted(HELD_OUT_ENVS):
        test_unseen.extend(sorted(envs[env]))

    budget = seen_frac * sum(frames.values())

    # Round-robin one trajectory at a time across eligible environments, so the
    # frame budget buys coverage of as many distinct environments as possible
    # instead of being spent on a few long trajectories.
    pools = {env: rng.sample(trajs, len(trajs))
             for env, trajs in envs.items()
             if env not in HELD_OUT_ENVS and len(trajs) >= MIN_TRAJS_FOR_SEEN_TEST}
    order = sorted(pools)
    rng.shuffle(order)

    picked: set = set()
    spent = 0
    progressed = True
    while progressed and spent < budget:
        progressed = False
        for env in order:
            if spent >= budget:
                break
            if not pools[env]:
                continue
            traj = pools[env].pop()
            # Never let an environment fall below the eligibility threshold in train.
            if len(envs[env]) - sum(1 for t in picked if t.startswith(env + "_")) - 1 < MIN_TRAJS_FOR_SEEN_TEST - 1:
                continue
            picked.add(traj)
            spent += frames[traj]
            progressed = True

    held = picked | set(test_unseen)
    train = [t for env_trajs in envs.values() for t in env_trajs if t not in held]

    return sorted(train), sorted(picked), sorted(test_unseen)


def write_split(split_root: str, name: str, traj_names: List[str]) -> None:
    out_dir = os.path.join(split_root, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "traj_names.txt"), "w", encoding="utf-8") as f:
        for traj in traj_names:
            f.write(traj + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True,
                        help="Root of the raw TartanGround tree.")
    parser.add_argument("--split-root",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train", "vint_train",
                                             "data", "data_splits", "tartan_ground"),
                        help="Where to write <split>/traj_names.txt.")
    parser.add_argument("--seen-frac", type=float, default=DEFAULT_SEEN_FRAC,
                        help="Fraction of all frames held out into the seen-scene "
                             "test split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the split without writing files.")
    args = parser.parse_args()

    train, test, test_unseen = build(args.raw_root, args.seen_frac, args.seed)

    counts = {}
    for name, trajs in (("train", train), ("test", test), ("test_unseen", test_unseen)):
        counts[name] = sum(frame_count(args.raw_root, t) for t in trajs)

    total_trajs = len(train) + len(test) + len(test_unseen)
    total_frames = sum(counts.values())
    print(f"{'split':<14}{'trajs':>7}{'frames':>10}{'frames %':>10}")
    for name, trajs in (("train", train), ("test", test), ("test_unseen", test_unseen)):
        print(f"{name:<14}{len(trajs):>7}{counts[name]:>10}"
              f"{100.0 * counts[name] / total_frames:>9.1f}%")
    print(f"{'TOTAL':<14}{total_trajs:>7}{total_frames:>10}")
    print()
    env_of = lambda names: {t.rsplit("_", 1)[0] for t in names}
    print(f"environments: train={len(env_of(train))}, "
          f"test covers {len(env_of(test))} of them, "
          f"test_unseen={len(env_of(test_unseen))} held out")
    print("held-out environments:")
    for env, kind in sorted(HELD_OUT_ENVS.items()):
        n = len([t for t in test_unseen if t.startswith(env + "_")])
        print(f"  {env:<16} ({kind}, {n} trajs)")

    if args.dry_run:
        print("\n[dry run] no files written")
        return

    for name, trajs in (("train", train), ("test", test), ("test_unseen", test_unseen)):
        write_split(args.split_root, name, trajs)
    print(f"\nwrote splits to {args.split_root}/{{train,test,test_unseen}}/traj_names.txt")


if __name__ == "__main__":
    main()
