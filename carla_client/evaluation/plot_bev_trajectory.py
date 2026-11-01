import argparse
import csv
import json
import math
import os
import queue
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import carla


def build_episode_colors(episodes: List[int]):
    episode_ids = sorted(episodes)
    if not episode_ids:
        return {}

    if len(episode_ids) <= 20:
        cmap = plt.get_cmap("tab20", len(episode_ids))
        return {ep: cmap(idx) for idx, ep in enumerate(episode_ids)}

    cmap = plt.get_cmap("hsv", len(episode_ids))
    return {ep: cmap(idx) for idx, ep in enumerate(episode_ids)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render true CARLA top-down RGB map image and overlay recorded trajectory."
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--metadata_filename", type=str, default="metadata.json")
    parser.add_argument("--trajectory_filename", type=str, default="trajectory.csv")
    parser.add_argument("--output", type=str, default="bev_plot.png")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--margin_m", type=float, default=30.0)
    parser.add_argument("--episode", type=int, default=-1, help="-1 means all episodes")
    parser.add_argument("--cam_loc", type=str, default=None, help="Override camera location as 'x,y,z'")
    parser.add_argument(
        "--max_step",
        type=float,
        default=-1,
        help=(
            "Maximum step_idx to include (inclusive). "
            "If 0 < value < 1, treat as a fraction of each episode's max step_idx. "
            "-1 means all steps."
        ),
    )
    parser.add_argument(
        "--cell_size_m",
        type=float,
        default=4.0,
        help="Grid cell size in meters used for explored area calculation.",
    )
    parser.add_argument(
        "--interp_step_m",
        type=float,
        default=0.5,
        help="Sampling step along trajectory segments in meters for coverage computation.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="exploration_metrics.json",
        help="Optional output JSON path for exploration metrics.",
    )
    return parser.parse_args()


def load_metadata(save_dir: str, metadata_filename: str) -> Dict:
    with open(os.path.join(save_dir, metadata_filename), "r", encoding="utf-8") as f:
        return json.load(f)


def load_rows(save_dir: str, trajectory_filename: str):
    trajectory_path = os.path.abspath(os.path.join(save_dir, trajectory_filename))
    print(f"[TRAJ] Using trajectory CSV: {trajectory_path}")

    rows_by_episode: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    with open(trajectory_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_episode[int(row["episode_idx"])].append(row)
    for ep in rows_by_episode:
        rows_by_episode[ep].sort(key=lambda r: int(r["step_idx"]))
    return rows_by_episode

def filter_rows_by_max_step(rows_by_episode: Dict[int, List[Dict[str, str]]], max_step: float):
    if max_step < 0:
        return rows_by_episode

    total_steps = sum(len(rows) for rows in rows_by_episode.values())
    if 0 < max_step < 1:
        target_steps = int(math.floor(total_steps * max_step))
    else:
        target_steps = int(math.floor(max_step))

    filtered: Dict[int, List[Dict[str, str]]] = {}
    steps_retained = 0

    for ep in sorted(rows_by_episode.keys()):
        rows = rows_by_episode[ep]
        if steps_retained >= target_steps:
            break

        allowed = target_steps - steps_retained
        limited_rows = rows[:allowed]

        if limited_rows:
            filtered[ep] = limited_rows

        steps_retained += len(limited_rows)

    return filtered


def get_points(rows_by_episode: Dict[int, List[Dict[str, str]]], episode: int):
    points = []
    for ep, rows in rows_by_episode.items():
        if episode >= 0 and ep != episode:
            continue
        for r in rows:
            points.append((float(r["x"]), float(r["y"]), float(r["z"]), ep))
    if not points:
        raise RuntimeError("No trajectory points found for selected episode.")
    return points


def build_camera(width: int, height: int, fov_deg: float):
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    k = np.array(
        [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    return k


def project_world_to_pixel(
    location: carla.Location,
    world_to_camera: np.ndarray,
    k: np.ndarray,
):
    p = np.array([location.x, location.y, location.z, 1.0], dtype=np.float64)
    p_cam = world_to_camera @ p
    # CARLA -> standard camera axis mapping.
    x = p_cam[1]
    y = -p_cam[2]
    z = p_cam[0]
    if z <= 1e-3:
        return None
    u = (k[0, 0] * x / z) + k[0, 2]
    v = (k[1, 1] * y / z) + k[1, 2]
    return float(u), float(v)


def capture_topdown(world: carla.World, cam_tf: carla.Transform, image_size: int, fov_deg: float):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(image_size))
    bp.set_attribute("image_size_y", str(image_size))
    bp.set_attribute("fov", str(fov_deg))
    bp.set_attribute("sensor_tick", "0.0")
    sensor = world.spawn_actor(bp, cam_tf)
    q: queue.Queue = queue.Queue()
    sensor.listen(q.put)
    image = None
    try:
        for _ in range(60):
            if world.get_settings().synchronous_mode:
                world.tick()
            else:
                world.wait_for_tick(1.0)
            try:
                image = q.get_nowait()
                break
            except queue.Empty:
                pass
        if image is None:
            raise RuntimeError("Timed out waiting for RGB sensor frame.")
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )
        return arr[:, :, :3][:, :, ::-1], sensor
    except Exception:
        sensor.destroy()
        raise


def _sample_segment_2d(p0: Tuple[float, float], p1: Tuple[float, float], step_m: float):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    segment_length = math.hypot(dx, dy)
    if segment_length < 1e-9:
        return [(x0, y0)]
    num_steps = max(1, int(math.ceil(segment_length / max(step_m, 1e-3))))
    return [(x0 + dx * (i / num_steps), y0 + dy * (i / num_steps)) for i in range(num_steps + 1)]


def compute_exploration_metrics(
    rows_by_episode: Dict[int, List[Dict[str, str]]],
    episode: int,
    cell_size_m: float,
    interp_step_m: float,
):
    selected_eps = sorted(rows_by_episode.keys()) if episode < 0 else [episode]
    per_episode: Dict[int, Dict[str, float]] = {}
    all_unique_cells = set()
    total_path_m = 0.0

    for ep in selected_eps:
        rows = rows_by_episode.get(ep, [])
        if len(rows) < 2:
            per_episode[ep] = {
                "path_length_m": 0.0,
                "new_area_m2": 0.0,
                "area_per_meter": 0.0,
            }
            continue

        xy = [(float(r["x"]), float(r["y"])) for r in rows]
        unique_cells = set()
        path_length_m = 0.0

        for idx in range(1, len(xy)):
            x0, y0 = xy[idx - 1]
            x1, y1 = xy[idx]
            path_length_m += math.hypot(x1 - x0, y1 - y0)

            for sx, sy in _sample_segment_2d((x0, y0), (x1, y1), interp_step_m):
                cell = (int(math.floor(sx / cell_size_m)), int(math.floor(sy / cell_size_m)))
                unique_cells.add(cell)
                all_unique_cells.add(cell)

        new_area_m2 = len(unique_cells) * (cell_size_m ** 2)
        per_episode[ep] = {
            "path_length_m": path_length_m,
            "new_area_m2": new_area_m2,
            "area_per_meter": (new_area_m2 / path_length_m) if path_length_m > 1e-9 else 0.0,
        }
        total_path_m += path_length_m

    total_area_m2 = len(all_unique_cells) * (cell_size_m ** 2)
    aggregate = {
        "path_length_m": total_path_m,
        "new_area_m2": total_area_m2,
        "area_per_meter": (total_area_m2 / total_path_m) if total_path_m > 1e-9 else 0.0,
    }
    return {"per_episode": per_episode, "aggregate": aggregate}


def main():
    args = parse_args()
    save_dir = os.path.abspath(args.save_dir)
    metadata = load_metadata(save_dir, args.metadata_filename)
    rows_by_episode = load_rows(save_dir, args.trajectory_filename)
    rows_by_episode = filter_rows_by_max_step(rows_by_episode, args.max_step)
    if args.max_step >= 0:
        if 0 < args.max_step < 1:
            print(
                "[TRAJ] Applying global fractional step limit: "
                f"retaining {args.max_step * 100}% of total steps."
            )
        else:
            print(f"[TRAJ] Applying global step limit: retaining {int(math.floor(args.max_step))} steps total")
    points = get_points(rows_by_episode, args.episode)

    metrics = compute_exploration_metrics(
        rows_by_episode,
        args.episode,
        cell_size_m=args.cell_size_m,
        interp_step_m=args.interp_step_m,
    )
    aggregate = metrics["aggregate"]
    print(
        "[METRICS] Exploration quality: "
        f"path_length_m={aggregate['path_length_m']:.2f}, "
        f"new_area_m2={aggregate['new_area_m2']:.2f}, "
        f"coverage_efficiency_m2_per_m={aggregate['area_per_meter']:.4f}"
    )
    if args.metrics_out:
        metrics_out = os.path.abspath(os.path.join(args.save_dir, args.metrics_out))
        os.makedirs(os.path.dirname(metrics_out) or ".", exist_ok=True)
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[METRICS] Saved metrics JSON: {metrics_out}")

    if args.cam_loc is None:

        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        zs = np.array([p[2] for p in points], dtype=np.float64)
        # cx, cy = float(xs.mean()), float(ys.mean())
        cx = (xs.max() + xs.min()) / 2.0
        cy = (ys.max() + ys.min()) / 2.0
        span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) + 2.0 * args.margin_m
        z_base = float(np.percentile(zs, 95))

        # For downward camera, ground half-width ~= h * tan(fov/2)
        height = max(40.0, span / (2.0 * np.tan(np.radians(args.fov_deg) / 2.0)))
        cam_tf = carla.Transform(
            carla.Location(x=cx, y=cy, z=z_base + height),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
    else:
        cx, cy, cz = args.cam_loc.split(",")
        cx, cy, cz = float(cx), float(cy), float(cz)
        cam_tf = carla.Transform(
            carla.Location(x=cx, y=cy, z=cz),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

    print(f"[BEV] Capturing top-down view from camera at "
          f"location=({cam_tf.location.x:.1f}, {cam_tf.location.y:.1f}, {cam_tf.location.z:.1f})")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    wanted_world = metadata.get("world_name")
    world = client.load_world(wanted_world) if wanted_world else client.get_world()

    rgb, sensor = capture_topdown(world, cam_tf, args.image_size, args.fov_deg)
    try:
        world_to_camera = np.array(sensor.get_transform().get_inverse_matrix(), dtype=np.float64)
        k = build_camera(args.image_size, args.image_size, args.fov_deg)

        uv_by_ep: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        # lhc_uv: List[Tuple[float, float]] = []
        for x, y, z, ep in points:
            uv = project_world_to_pixel(
                carla.Location(x=float(x), y=float(y), z=float(z) + 0.3),
                world_to_camera,
                k,
            )
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < args.image_size and 0 <= v < args.image_size:
                uv_by_ep[ep].append((u, v))

        # for ep, rows in rows_by_episode.items():
        #     if args.episode >= 0 and ep != args.episode:
        #         continue
        #     for row in rows:
        #         if row["lhc_flag"].strip().lower() not in {"true", "1"}:
        #             continue
        #         uv = project_world_to_pixel(
        #             carla.Location(
        #                 x=float(row["x"]),
        #                 y=float(row["y"]),
        #                 z=float(row["z"]) + 0.3,
        #             ),
        #             world_to_camera,
        #             k,
        #         )
        #         if uv is None:
        #             continue
        #         u, v = uv
        #         if 0 <= u < args.image_size and 0 <= v < args.image_size:
        #             lhc_uv.append((u, v))

        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        ax.imshow(rgb)
        # ax.set_title(f"Top-Down RGB + Trajectory ({world.get_map().name})")
        # episode_colors = build_episode_colors(list(uv_by_ep.keys()))
        
        all_u, all_v = [], []
        for ep in sorted(uv_by_ep.keys()):
            if args.episode >= 0 and ep != args.episode:
                continue
            uv = uv_by_ep[ep]
            if len(uv) < 2:
                continue
            u = [p[0] for p in uv]
            v = [p[1] for p in uv]
            # c = episode_colors[ep]
            # ax.plot(u, v, color=c, linewidth=5.0, label=f"ep {ep}", zorder=2)
            # ax.scatter(u[0], v[0], color=c, marker="o", s=20, zorder=3)
            # ax.scatter(u[-1], v[-1], color=c, marker="x", s=20, zorder=3)
            ax.plot(u, v, color="red", linewidth=7.0, zorder=2)
            all_u.extend(u)
            all_v.extend(v)
            
        if all_u and all_v:
            ax.scatter(all_u[0], all_v[0], color="darkgreen", marker="o", s=100, zorder=3)
            ax.scatter(all_u[-1], all_v[-1], color="darkblue", marker="o", s=100, zorder=3)
            
        # if lhc_uv:
        #     lhc_u = [p[0] for p in lhc_uv]
        #     lhc_v = [p[1] for p in lhc_uv]
        #     ax.scatter(lhc_u, lhc_v, color="black", marker="o", s=24, linewidths=0, zorder=4)
        ax.set_axis_off()
        # if args.episode < 0 and len(uv_by_ep) <= 10:
        #     ax.legend(loc="upper right")
        out = os.path.abspath(os.path.join(args.save_dir, args.output))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"[BEV] Saved rendered overlay: {out}")
    finally:
        sensor.stop()
        sensor.destroy()


if __name__ == "__main__":
    main()
