#!/usr/bin/env python3
"""
Convert the TartanGround subset into the canonical trajectory format used by
ViNT_Dataset.

Input (per trajectory):
    <raw_root>/<Env>/Data_diff/<P1xxx>/
        image_lcam_front/%06d_lcam_front.png     640x640 RGB PNG
        pose_lcam_front.txt                      N lines of "tx ty tz qx qy qz qw"
        <P1xxx>_metadata.json

Output (per trajectory):
    <out_root>/<Env>_<P1xxx>/
        0.jpg, 1.jpg, ...                        640x480 JPEG (4:3)
        traj_data.pkl                            {"position": (N,2), "yaw": (N,)}
        metadata.json                            copied through unchanged

Coordinate frames
-----------------
TartanGround poses are in a NED world frame (x=North, y=East, z=Down) with the
camera body frame x-forward / y-right / z-down. That is right-handed with yaw
about the down-axis rotating x toward y, which is exactly the convention
`yaw_rotmat` and `to_local_coords` in vint_train.data.data_utils assume. So the
planar position is taken as-is from (tx, ty) and no axis swap or sign flip is
applied. Yaw is the first ZYX Euler angle of the pose quaternion, which is
scalar-last (xyzw) -- the ordering scipy expects by default.

The z (Down) component is dropped: the training pipeline is planar. On sloped
terrain this understates true travelled distance.

Image handling
--------------
Source frames are 640x640. `resize_and_aspect_crop` center-crops to a 4:3 aspect
ratio at load time, so this script applies exactly that crop once (640x640 ->
640x480) and stores the result. Re-cropping a 640x480 image to 4:3 at load time
is a no-op, so the pixels the model sees are identical either way.
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation

IMAGE_ASPECT_RATIO = 4 / 3


def center_crop(img: Image.Image, aspect_ratio: float = IMAGE_ASPECT_RATIO) -> Image.Image:
    """Center-crop to `aspect_ratio`, matching data_utils.resize_and_aspect_crop."""
    w, h = img.size
    if w > h:
        th, tw = h, int(h * aspect_ratio)
    else:
        th, tw = int(w / aspect_ratio), w
    left, top = (w - tw) // 2, (h - th) // 2
    return img.crop((left, top, left + tw, top + th))


def poses_to_traj_data(pose_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read pose_lcam_front.txt -> (position (N,2) float64, yaw (N,) float64)."""
    poses = np.loadtxt(pose_path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] != 7:
        raise ValueError(f"{pose_path}: expected 7 columns, got {poses.shape[1]}")

    position = poses[:, :2].copy()                      # NED (North, East)
    # Quaternions are scalar-last (xyzw); ZYX Euler -> first angle is yaw.
    yaw = Rotation.from_quat(poses[:, 3:]).as_euler("ZYX")[:, 0]
    return position, yaw


def convert_traj(
    raw_traj_dir: str,
    out_root: str,
    quality: int,
    resize: Optional[Tuple[int, int]],
    overwrite: bool,
) -> Tuple[str, int, Optional[str]]:
    """Convert one trajectory. Returns (traj_name, n_frames, error_or_None)."""
    env = raw_traj_dir.split(os.sep)[-3]
    traj_id = os.path.basename(raw_traj_dir)
    traj_name = f"{env}_{traj_id}"
    out_dir = os.path.join(out_root, traj_name)

    try:
        pose_path = os.path.join(raw_traj_dir, "pose_lcam_front.txt")
        position, yaw = poses_to_traj_data(pose_path)

        frames = sorted(glob.glob(os.path.join(raw_traj_dir, "image_lcam_front", "*.png")))
        if len(frames) != len(position):
            raise ValueError(
                f"{len(frames)} images but {len(position)} poses -- refusing to guess alignment"
            )

        os.makedirs(out_dir, exist_ok=True)

        for i, src in enumerate(frames):
            dst = os.path.join(out_dir, f"{i}.jpg")
            if not overwrite and os.path.exists(dst):
                continue
            img = Image.open(src).convert("RGB")
            img = center_crop(img)
            if resize is not None:
                img = img.resize(resize, Image.BILINEAR)
            # Write to a temp name then rename, so an interrupted run never
            # leaves a truncated JPEG that a resumed run would skip over.
            tmp = dst + ".tmp"
            img.save(tmp, format="JPEG", quality=quality)
            os.replace(tmp, dst)

        traj_data = {"position": position, "yaw": yaw}
        with open(os.path.join(out_dir, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)

        meta_src = os.path.join(raw_traj_dir, f"{traj_id}_metadata.json")
        if os.path.exists(meta_src):
            shutil.copyfile(meta_src, os.path.join(out_dir, "metadata.json"))

        return traj_name, len(frames), None
    except Exception as exc:  # noqa: BLE001 - reported per trajectory, run continues
        return traj_name, 0, f"{type(exc).__name__}: {exc}"


def find_trajs(raw_root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(raw_root, "*", "Data_diff", "P*")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality (default 90).")
    parser.add_argument("--resize", default=None, metavar="WxH",
                        help="Optional resize after the 4:3 crop, e.g. 320x240. "
                             "Default keeps native 640x480.")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode frames that already exist (default: resume).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Convert only the first N trajectories (for a smoke test).")
    args = parser.parse_args()

    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    trajs = find_trajs(args.raw_root)
    if args.limit:
        trajs = trajs[: args.limit]
    if not trajs:
        raise SystemExit(f"No trajectories found under {args.raw_root}")

    os.makedirs(args.out_root, exist_ok=True)
    print(f"converting {len(trajs)} trajectories -> {args.out_root} "
          f"({args.workers} workers, quality={args.quality}, "
          f"resize={'native 640x480' if resize is None else resize})")

    total_frames, failures = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(convert_traj, t, args.out_root, args.quality, resize, args.overwrite)
            for t in trajs
        ]
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
            name, n, err = fut.result()
            if err:
                failures.append((name, err))
            else:
                total_frames += n

    print(f"\nconverted {len(trajs) - len(failures)}/{len(trajs)} trajectories, "
          f"{total_frames} frames")
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for name, err in failures:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
