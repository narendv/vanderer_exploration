import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import carla

from evaluation.plot_bev_trajectory import (
    build_camera,
    capture_topdown,
    load_metadata,
    load_rows,
    project_world_to_pixel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a BEV trajectory video that grows with each step."
    )
    parser.add_argument("--town", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=str, default="select")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override directory containing metadata.json and trajectory.csv",
    )
    parser.add_argument("--metadata_filename", type=str, default="metadata.json")
    parser.add_argument("--trajectory_filename", type=str, default="trajectory.csv")
    parser.add_argument(
        "--cam_loc",
        type=str,
        required=True,
        help="Camera location as 'x,y,z'",
    )
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=str,
        default="bev_progress.mp4",
        help="Video filename (relative to save_dir unless absolute)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--step_stride", type=int, default=50)
    parser.add_argument("--line_width", type=int, default=6)
    parser.add_argument("--head_radius", type=int, default=6)
    parser.add_argument(
        "--episode",
        type=int,
        default=-1,
        help="Episode index to render (-1 means all episodes)",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=-1,
        help="Maximum global step index to include (-1 means all steps)",
    )
    return parser.parse_args()


def resolve_save_dir(args: argparse.Namespace) -> str:
    if args.save_dir:
        return os.path.abspath(args.save_dir)
    folder_name = f"recorded_trajs_town_{args.town}_seed_{args.seed}"
    return os.path.abspath(os.path.join(args.base_dir, args.mode, folder_name))


def parse_cam_loc(cam_loc: str) -> carla.Transform:
    parts = [p.strip() for p in cam_loc.split(",")]
    if len(parts) != 3:
        raise ValueError("cam_loc must be in the form 'x,y,z'")
    x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )


def flatten_rows(
    rows_by_episode: Dict[int, List[Dict[str, str]]],
    episode: int,
) -> List[Tuple[Dict[str, str], int]]:
    ordered: List[Tuple[Dict[str, str], int]] = []
    for ep in sorted(rows_by_episode.keys()):
        if episode >= 0 and ep != episode:
            continue
        for row in rows_by_episode[ep]:
            ordered.append((row, ep))
    return ordered


def project_rows_to_uv(
    ordered_rows: List[Tuple[Dict[str, str], int]],
    world_to_camera: np.ndarray,
    k: np.ndarray,
    image_size: int,
) -> List[Optional[Tuple[int, int]]]:
    uv_list: List[Optional[Tuple[int, int]]] = []
    for row, _ in ordered_rows:
        uv = project_world_to_pixel(
            carla.Location(
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]) + 0.3,
            ),
            world_to_camera,
            k,
        )
        if uv is None:
            uv_list.append(None)
            continue
        u, v = uv
        if 0 <= u < image_size and 0 <= v < image_size:
            uv_list.append((int(round(u)), int(round(v))))
        else:
            uv_list.append(None)
    return uv_list


def main() -> None:
    args = parse_args()
    save_dir = resolve_save_dir(args)

    metadata = load_metadata(save_dir, args.metadata_filename)
    rows_by_episode = load_rows(save_dir, args.trajectory_filename)
    ordered_rows = flatten_rows(rows_by_episode, args.episode)
    if not ordered_rows:
        raise RuntimeError("No trajectory rows found for the requested selection.")

    if args.max_step >= 0:
        ordered_rows = ordered_rows[: args.max_step + 1]

    cam_tf = parse_cam_loc(args.cam_loc)
    print(
        "[BEV] Using camera at "
        f"({cam_tf.location.x:.1f}, {cam_tf.location.y:.1f}, {cam_tf.location.z:.1f})"
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    wanted_world = metadata.get("world_name")
    world = client.load_world(wanted_world) if wanted_world else client.get_world()

    rgb = None
    sensor = None
    video_writer = None
    try:
        rgb, sensor = capture_topdown(world, cam_tf, args.image_size, args.fov_deg)
        world_to_camera = np.array(sensor.get_transform().get_inverse_matrix(), dtype=np.float64)
        k = build_camera(args.image_size, args.image_size, args.fov_deg)

        uv_list = project_rows_to_uv(ordered_rows, world_to_camera, k, args.image_size)
        if not any(p is not None for p in uv_list):
            raise RuntimeError("No trajectory points project into the camera view.")

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV (cv2) is required to write the video.") from exc

        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(save_dir, output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            float(args.fps),
            (args.image_size, args.image_size),
        )

        background_bgr = rgb[:, :, ::-1].copy()
        path_img = background_bgr.copy()
        prev_uv: Optional[Tuple[int, int]] = None
        start_uv: Optional[Tuple[int, int]] = None

        total_steps = len(uv_list)
        stride = max(1, int(args.step_stride))
        log_every = max(1, stride * 50)

        for idx, uv in enumerate(uv_list):
            if uv is not None:
                if start_uv is None:
                    start_uv = uv
                    cv2.circle(path_img, start_uv, args.head_radius + 2, (0, 180, 0), -1, lineType=cv2.LINE_AA)
                if prev_uv is not None:
                    cv2.line(path_img, prev_uv, uv, (0, 0, 255), args.line_width, lineType=cv2.LINE_AA)
                prev_uv = uv
            else:
                prev_uv = None

            if (idx % stride == 0) or (idx == total_steps - 1):
                frame = path_img.copy()
                if uv is not None:
                    cv2.circle(frame, uv, args.head_radius, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                video_writer.write(frame)

            if idx % log_every == 0:
                print(f"[BEV] Progress: step {idx + 1}/{total_steps}")

        print(f"[BEV] Video saved to {output_path}")

    finally:
        if video_writer is not None:
            video_writer.release()
        if sensor is not None:
            sensor.stop()
            sensor.destroy()


if __name__ == "__main__":
    main()
