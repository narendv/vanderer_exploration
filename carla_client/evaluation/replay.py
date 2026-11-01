import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import carla
import gymnasium as gym
import numpy as np

import envs
from envs.carla.utils import preproc_carla_img


def set_birdeye_spectator(world: carla.World, ego_loc: carla.Location, height_m: float) -> None:
    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=float(ego_loc.x), y=float(ego_loc.y), z=float(ego_loc.z + height_m)),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
    )


def transform_from_dict(data: Dict) -> carla.Transform:
    location = data["location"]
    rotation = data["rotation"]
    return carla.Transform(
        carla.Location(
            x=float(location["x"]),
            y=float(location["y"]),
            z=float(location["z"]),
        ),
        carla.Rotation(
            pitch=float(rotation["pitch"]),
            yaw=float(rotation["yaw"]),
            roll=float(rotation["roll"]),
        ),
    )


def transform_from_row(row: Dict[str, str]) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
        ),
        carla.Rotation(
            pitch=float(row["pitch"]),
            yaw=float(row["yaw"]),
            roll=float(row["roll"]),
        ),
    )


def lifted_location(loc: carla.Location, z_offset: float = 0.35) -> carla.Location:
    return carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z + z_offset))


def parse_args():
    parser = argparse.ArgumentParser(description="Replay custom NOMAD trajectory recording")
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory containing metadata.json and trajectory.csv",
    )
    parser.add_argument("--metadata_filename", type=str, default="metadata.json")
    parser.add_argument(
        "--trajectory_filename",
        type=str,
        default="trajectory.csv",
        help="Used if metadata does not include trajectory_path",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=0,
        help="0 means replay all logged steps for each episode",
    )
    parser.add_argument(
        "--bird_height_m",
        type=float,
        default=80.0,
        help="Top-down spectator height above ego vehicle",
    )
    parser.add_argument(
        "--draw_path",
        action="store_true",
        help="Draw recorded trajectory path in-world for visualization",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record the attached RGB camera view to a video",
    )
    parser.add_argument(
        "--video_out",
        type=str,
        default="replay_camera.mp4",
        help="Output video filename (relative to save_dir unless absolute)",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=None,
        help="Frames per second (defaults to 1/dt if unset)",
    )
    parser.add_argument(
        "--video_stride",
        type=int,
        default=1,
        help="Write every Nth step to the video",
    )
    parser.add_argument(
        "--global_stride",
        type=int,
        default=0,
        help="If >0, replay and record only rows whose global index (across episodes) "
        "is a multiple of N, plus the final row - frame-matches "
        "plot_bev_trajectory_video.py --step_stride N",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Launch CARLA with offscreen rendering",
    )
    return parser.parse_args()


def load_metadata(save_dir: str, metadata_filename: str) -> Dict:
    metadata_path = os.path.join(save_dir, metadata_filename)
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rows(trajectory_path: str) -> Dict[int, List[Dict[str, str]]]:
    rows_by_episode: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    with open(trajectory_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode_idx = int(row["episode_idx"])
            rows_by_episode[episode_idx].append(row)
    for episode_idx in rows_by_episode:
        rows_by_episode[episode_idx].sort(key=lambda x: int(x["step_idx"]))
    return rows_by_episode


def fetch_latest_camera_rgb(base_env) -> Optional[np.ndarray]:
    if not base_env.sensors:
        return None
    buffer = base_env.sensors[0]["buffer"]
    latest = None
    while buffer:
        latest = buffer.popleft()
    if latest is None:
        return None
    rgb = preproc_carla_img(latest, channel_first=False)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb_u8


def main():
    args = parse_args()
    if args.offscreen:
        os.environ["CARLA_OFFSCREEN"] = "1"
    save_dir = os.path.abspath(args.save_dir)
    metadata = load_metadata(save_dir, args.metadata_filename)

    trajectory_path = metadata.get("trajectory_path")
    if not trajectory_path:
        trajectory_path = os.path.join(save_dir, args.trajectory_filename)
    if not os.path.isabs(trajectory_path):
        trajectory_path = os.path.join(save_dir, trajectory_path)
    trajectory_path = os.path.abspath(trajectory_path)

    rows_by_episode = load_rows(trajectory_path)
    if not rows_by_episode:
        raise RuntimeError(f"No recorded rows found in {trajectory_path}")

    town = int(metadata["town"])
    base_seed = int(metadata["seed"])
    env = gym.make(f"carla-town{town:02d}-v1")
    base_env = env.unwrapped

    video_writer = None
    video_path = None
    cv2_module = None
    if args.record_video:
        try:
            import cv2 as cv2_module  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV (cv2) is required to record video.") from exc

        video_path = args.video_out
        if not os.path.isabs(video_path):
            video_path = os.path.join(save_dir, video_path)
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    episode_starts = {
        int(e["episode_idx"]): e for e in metadata.get("episode_starts", [])
    }

    print(f"[REPLAY] save_dir={save_dir}")
    print(f"[REPLAY] town={town} trajectory={trajectory_path}")
    print(f"[REPLAY] episodes={len(rows_by_episode)}")

    total_rows = sum(len(r) for r in rows_by_episode.values())
    global_offset = 0

    try:
        for episode_idx in sorted(rows_by_episode.keys()):
            rows = rows_by_episode[episode_idx]
            if len(rows) == 0:
                continue

            start_entry: Optional[Dict] = episode_starts.get(episode_idx)
            spawn_point = None
            reset_seed = base_seed
            if start_entry is not None:
                spawn_point = transform_from_dict(start_entry["spawn_transform"])
                reset_seed = base_seed

            _, _ = env.reset(
                return_info=True,
                seed=reset_seed,
                spawn_noise_std=0.0,
                action_exec_noise_std=0.0,
                spawn_point=spawn_point,
            )
            steps = 0
            max_steps = args.max_steps_per_episode if args.max_steps_per_episode > 0 else len(rows)

            print(
                f"[REPLAY] episode={episode_idx} seed={reset_seed} "
                f"logged_steps={len(rows)} replay_steps={max_steps}"
            )

            if args.draw_path and len(rows) >= 2:
                for i in range(1, len(rows)):
                    p0_raw = transform_from_row(rows[i - 1]).location
                    p1_raw = transform_from_row(rows[i]).location
                    if p0_raw.distance(p1_raw) < 0.05:
                        continue
                    p0 = lifted_location(p0_raw)
                    p1 = lifted_location(p1_raw)
                    base_env.world.debug.draw_line(
                        p0,
                        p1,
                        thickness=1.0,
                        color=carla.Color(255, 0, 0),
                        life_time=0.0,
                    )

            while steps < max_steps:
                if args.global_stride > 0:
                    g = global_offset + steps
                    if g % args.global_stride != 0 and g != total_rows - 1:
                        steps += 1
                        continue
                row = rows[steps]
                tf = transform_from_row(row)
                base_env.vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                base_env.vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                base_env.vehicle.set_transform(tf)
                base_env.world.tick()
                set_birdeye_spectator(base_env.world, tf.location, args.bird_height_m)

                if args.record_video and (
                    args.global_stride > 0 or steps % max(1, args.video_stride) == 0
                ):
                    frame = fetch_latest_camera_rgb(base_env)
                    if frame is not None:
                        if video_writer is None:
                            fps = args.video_fps if args.video_fps is not None else 1.0 / base_env.dt
                            height, width = frame.shape[:2]
                            fourcc = cv2_module.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2_module.VideoWriter(
                                video_path,
                                fourcc,
                                float(fps),
                                (width, height),
                            )
                        frame_bgr = frame[:, :, ::-1]
                        video_writer.write(frame_bgr)
                steps += 1
            global_offset += len(rows)

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[REPLAY] Video saved to {video_path}")
        env.close()
        print("[REPLAY] done")


if __name__ == "__main__":
    main()
