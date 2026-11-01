import argparse
import json
import math
import pickle
import queue
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import carla


TOWN_TO_WORLD = {
    1: "Town01_Opt",
    2: "Town02_Opt",
    3: "Town03_Opt",
    4: "Town04_Opt",
    5: "Town05_Opt",
    # Towns 06/07/11/12 have no "_Opt" map variant.
    6: "Town06",
    7: "Town07",
    10: "Town10HD_Opt",
    11: "Town11",
    12: "Town12",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render CARLA BEV map and trajectory coverage heatmap from generated traj_data.pkl dataset."
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Root directory with rollout subdirs")
    parser.add_argument(
        "--towns",
        type=str,
        default="",
        help="Optional comma-separated town IDs (e.g. 4,10). Default: all towns found.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--margin_m", type=float, default=30.0)
    parser.add_argument(
        "--interp_step_m",
        type=float,
        default=0.5,
        help="Step size (meters) to sample along trajectory segments before projection.",
    )
    parser.add_argument(
        "--max_rollouts_per_town",
        type=int,
        default=0,
        help="0 means use all rollouts. Otherwise cap rollouts per town.",
    )
    parser.add_argument(
        "--min_traj_length_m",
        type=float,
        default=0.0,
        help="Skip trajectories whose total path length (meters) is below this value.",
    )
    parser.add_argument(
        "--min_traj_frames",
        type=int,
        default=50,
        help="Skip trajectories with fewer than this many recorded frames (poses). 0 disables.",
    )
    parser.add_argument(
        "--min_straightness",
        type=float,
        default=0.0,
        help="Skip looping/circular trajectories whose net-displacement / path-length "
        "ratio is below this value (0..1). 0 disables. Try ~0.3 to drop circles.",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only plot trajectories listed in the per-town training split file.",
    )
    parser.add_argument(
        "--train_split_template",
        type=str,
        default="../train/vint_train/data/data_splits/carla_town{town}/train/traj_names.txt",
        help="Path template for the train split list. '{town}' is replaced by the town id.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="points",
        choices=["points", "heatmap"],
        help="Render trajectory coverage as scattered points (default) or a density heatmap.",
    )
    parser.add_argument(
        "--heatmap_alpha",
        type=float,
        default=0.65,
        help="Overlay alpha for heatmap layer.",
    )
    parser.add_argument(
        "--brush_radius_px",
        type=int,
        default=0.5,
        help="Radius in pixels used to thicken coverage marks in the heatmap.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.0,
        help="Marker size for scattered points (mode=points).",
    )
    parser.add_argument(
        "--point_color",
        type=str,
        default="blue",
        help="Marker color for scattered points (mode=points).",
    )
    parser.add_argument(
        "--point_alpha",
        type=float,
        default=0.5,
        help="Marker alpha for scattered points (mode=points).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Where output PNGs are written. Default: <dataset_dir>/heatmaps",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="dataset_heatmap",
        help="Output file prefix. Final name: <prefix>_townXX.png",
    )
    return parser.parse_args()


def parse_towns(towns_str: str) -> Optional[set]:
    if not towns_str.strip():
        return None
    towns = set()
    for token in towns_str.split(","):
        token = token.strip()
        if not token:
            continue
        town = int(token)
        if town not in TOWN_TO_WORLD:
            raise ValueError(f"Unsupported town {town}. Supported towns: {sorted(TOWN_TO_WORLD)}")
        towns.add(town)
    return towns if towns else None


def trajectory_length_m(positions_xy: np.ndarray) -> float:
    if positions_xy.shape[0] < 2:
        return 0.0
    deltas = np.diff(positions_xy[:, :2], axis=0)
    return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


def trajectory_straightness(positions_xy: np.ndarray) -> float:
    """Net start->end displacement divided by total path length.

    ~1.0 for a straight traverse, ~0.0 for a loop/circle that returns near its start.
    """
    path_len = trajectory_length_m(positions_xy)
    if path_len <= 1e-6:
        return 0.0
    net = float(np.hypot(*(positions_xy[-1, :2] - positions_xy[0, :2])))
    return net / path_len


def load_train_names(template: str, town: int) -> Optional[set]:
    """Read the per-town train split file. Returns a set of rollout names, or None if missing."""
    path = Path(template.format(town=town))
    if not path.exists():
        print(f"[WARN] town={town}: train split file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_rollouts(
    dataset_dir: Path,
    wanted_towns: Optional[set],
    min_traj_length_m: float = 0.0,
    min_traj_frames: int = 0,
    min_straightness: float = 0.0,
    train_split_template: Optional[str] = None,
) -> Dict[int, List[np.ndarray]]:
    by_town: Dict[int, List[np.ndarray]] = defaultdict(list)
    skipped_short = 0
    skipped_few_frames = 0
    skipped_circular = 0
    skipped_not_train = 0
    train_names_by_town: Dict[int, Optional[set]] = {}

    all_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    for rollout_dir in all_dirs:
        metadata_path = rollout_dir / "metadata.json"
        traj_path = rollout_dir / "traj_data.pkl"
        if not metadata_path.exists() or not traj_path.exists():
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            town = int(metadata["town"])
        except Exception as exc:
            print(f"[WARN] Skipping {rollout_dir}: failed to parse metadata ({exc})")
            continue

        if wanted_towns is not None and town not in wanted_towns:
            continue

        if train_split_template is not None:
            if town not in train_names_by_town:
                train_names_by_town[town] = load_train_names(train_split_template, town)
            train_names = train_names_by_town[town]
            if train_names is not None and rollout_dir.name not in train_names:
                skipped_not_train += 1
                continue

        try:
            with open(traj_path, "rb") as f:
                traj_data = pickle.load(f)
            positions_rh = np.asarray(traj_data["position"], dtype=np.float64)
        except Exception as exc:
            print(f"[WARN] Skipping {rollout_dir}: failed to read traj_data.pkl ({exc})")
            continue

        if positions_rh.ndim != 2 or positions_rh.shape[0] < 2 or positions_rh.shape[1] < 2:
            continue

        if min_traj_frames > 0 and positions_rh.shape[0] < min_traj_frames:
            skipped_few_frames += 1
            continue

        # Stored trajectory positions are right-handed. Convert back to CARLA's left-handed world.
        positions_lh = positions_rh[:, :2].copy()
        positions_lh[:, 1] *= -1.0

        if min_traj_length_m > 0.0 and trajectory_length_m(positions_lh) < min_traj_length_m:
            skipped_short += 1
            continue

        if min_straightness > 0.0 and trajectory_straightness(positions_lh) < min_straightness:
            skipped_circular += 1
            continue

        by_town[town].append(positions_lh)

    if skipped_not_train:
        print(f"[INFO] Skipped {skipped_not_train} trajectories not in the training split")
    if skipped_few_frames:
        print(f"[INFO] Skipped {skipped_few_frames} trajectories with fewer than {min_traj_frames} frames")
    if skipped_short:
        print(f"[INFO] Skipped {skipped_short} trajectories shorter than {min_traj_length_m:.1f} m")
    if skipped_circular:
        print(f"[INFO] Skipped {skipped_circular} circular/looping trajectories (straightness < {min_straightness:.2f})")

    return by_town


def build_camera(width: int, height: int, fov_deg: float):
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array(
        [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )


def project_world_to_pixel(location: carla.Location, world_to_camera: np.ndarray, k: np.ndarray):
    p = np.array([location.x, location.y, location.z, 1.0], dtype=np.float64)
    p_cam = world_to_camera @ p

    # CARLA camera axis -> standard camera axis
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
                continue

        if image is None:
            raise RuntimeError("Timed out waiting for top-down RGB frame.")

        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = arr[:, :, :3][:, :, ::-1]
        return rgb, sensor
    except Exception:
        sensor.destroy()
        raise


def sample_segment_2d(p0: Tuple[float, float], p1: Tuple[float, float], step_m: float):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return [(x0, y0)]

    n_steps = max(1, int(math.ceil(dist / max(step_m, 1e-3))))
    return [(x0 + dx * (i / n_steps), y0 + dy * (i / n_steps)) for i in range(n_steps + 1)]


def estimate_camera_transform(positions_xy: np.ndarray, map_spawn_z: float, fov_deg: float, margin_m: float):
    xs = positions_xy[:, 0]
    ys = positions_xy[:, 1]
    cx = float((xs.max() + xs.min()) * 0.5)
    cy = float((ys.max() + ys.min()) * 0.5)

    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) + (2.0 * margin_m)
    height = max(40.0, span / (2.0 * np.tan(np.radians(fov_deg) / 2.0)))

    return carla.Transform(
        carla.Location(x=cx, y=cy, z=float(map_spawn_z + height)),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )


def collect_raw_points(
    trajectories_xy: Iterable[np.ndarray],
    world_to_camera: np.ndarray,
    k: np.ndarray,
    image_size: int,
    z_world: float,
):
    """Project only the raw recorded poses (no segment interpolation)."""
    pixels = []
    for traj_xy in trajectories_xy:
        for i in range(traj_xy.shape[0]):
            uv = project_world_to_pixel(
                carla.Location(x=float(traj_xy[i, 0]), y=float(traj_xy[i, 1]), z=z_world),
                world_to_camera,
                k,
            )
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < image_size and 0 <= v < image_size:
                pixels.append((u, v))
    return pixels


def collect_projected_points(
    trajectories_xy: Iterable[np.ndarray],
    world_to_camera: np.ndarray,
    k: np.ndarray,
    image_size: int,
    z_world: float,
    interp_step_m: float,
):
    pixels = []
    for traj_xy in trajectories_xy:
        if traj_xy.shape[0] < 2:
            continue

        for i in range(1, traj_xy.shape[0]):
            p0 = (float(traj_xy[i - 1, 0]), float(traj_xy[i - 1, 1]))
            p1 = (float(traj_xy[i, 0]), float(traj_xy[i, 1]))
            for sx, sy in sample_segment_2d(p0, p1, interp_step_m):
                uv = project_world_to_pixel(
                    carla.Location(x=sx, y=sy, z=z_world),
                    world_to_camera,
                    k,
                )
                if uv is None:
                    continue
                u, v = uv
                if 0 <= u < image_size and 0 <= v < image_size:
                    pixels.append((u, v))
    return pixels


def render_heatmap(
    rgb: np.ndarray,
    pixels: List[Tuple[float, float]],
    alpha: float,
    brush_radius_px: int,
):
    h, w, _ = rgb.shape
    heat = np.zeros((h, w), dtype=np.float32)

    radius = max(0, int(brush_radius_px))
    if radius == 0:
        for u, v in pixels:
            iu = int(u)
            iv = int(v)
            heat[iv, iu] += 1.0
    else:
        offsets = []
        r2 = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (dx * dx + dy * dy) <= r2:
                    offsets.append((dx, dy))

        for u, v in pixels:
            iu = int(u)
            iv = int(v)
            for dx, dy in offsets:
                x = iu + dx
                y = iv + dy
                if 0 <= x < w and 0 <= y < h:
                    heat[y, x] += 1.0

    if heat.max() > 0:
        heat = np.log1p(heat)
        heat /= heat.max()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(rgb)
    ax.imshow(heat, cmap="hot", alpha=float(np.clip(alpha, 0.0, 1.0)))
    ax.set_axis_off()
    return fig


def render_points(
    rgb: np.ndarray,
    pixels: List[Tuple[float, float]],
    point_size: float,
    point_color: str,
    point_alpha: float,
):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(rgb)
    if pixels:
        us, vs = zip(*pixels)
        ax.scatter(
            us,
            vs,
            s=float(point_size),
            c=point_color,
            alpha=float(np.clip(point_alpha, 0.0, 1.0)),
            linewidths=0.0,
            edgecolors="none",
        )
    ax.set_axis_off()
    return fig


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else dataset_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_towns = parse_towns(args.towns)
    rollouts_by_town = load_rollouts(
        dataset_dir,
        wanted_towns,
        args.min_traj_length_m,
        args.min_traj_frames,
        args.min_straightness,
        args.train_split_template if args.train_only else None,
    )
    if not rollouts_by_town:
        raise RuntimeError("No valid trajectories found for requested towns.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)

    for town in sorted(rollouts_by_town.keys()):
        trajectories = rollouts_by_town[town]
        if args.max_rollouts_per_town > 0:
            trajectories = trajectories[: args.max_rollouts_per_town]

        world_name = TOWN_TO_WORLD.get(town)
        if world_name is None:
            print(f"[WARN] Skipping unsupported town={town}")
            continue

        print(f"[INFO] town={town} world={world_name} trajectories={len(trajectories)}")
        world = client.load_world(world_name)

        all_xy = np.concatenate(trajectories, axis=0)
        spawn_points = world.get_map().get_spawn_points()
        spawn_z = (
            float(np.median([sp.location.z for sp in spawn_points]))
            if spawn_points
            else 0.0
        )

        cam_tf = estimate_camera_transform(
            positions_xy=all_xy,
            map_spawn_z=spawn_z,
            fov_deg=args.fov_deg,
            margin_m=args.margin_m,
        )

        print(
            f"[INFO] town={town} camera=({cam_tf.location.x:.2f}, {cam_tf.location.y:.2f}, {cam_tf.location.z:.2f})"
        )

        rgb, sensor = capture_topdown(world, cam_tf, args.image_size, args.fov_deg)
        try:
            world_to_camera = np.array(sensor.get_transform().get_inverse_matrix(), dtype=np.float64)
            k = build_camera(args.image_size, args.image_size, args.fov_deg)

            if args.mode == "points":
                pixels = collect_raw_points(
                    trajectories_xy=trajectories,
                    world_to_camera=world_to_camera,
                    k=k,
                    image_size=args.image_size,
                    z_world=spawn_z + 0.3,
                )
            else:
                pixels = collect_projected_points(
                    trajectories_xy=trajectories,
                    world_to_camera=world_to_camera,
                    k=k,
                    image_size=args.image_size,
                    z_world=spawn_z + 0.3,
                    interp_step_m=args.interp_step_m,
                )

            if not pixels:
                print(f"[WARN] town={town}: no projected pixels inside frame, skipping output")
                continue

            if args.mode == "points":
                fig = render_points(
                    rgb=rgb,
                    pixels=pixels,
                    point_size=args.point_size,
                    point_color=args.point_color,
                    point_alpha=args.point_alpha,
                )
                title = f"Dataset Coverage Points - Town{town:02d}"
            else:
                fig = render_heatmap(
                    rgb=rgb,
                    pixels=pixels,
                    alpha=args.heatmap_alpha,
                    brush_radius_px=args.brush_radius_px,
                )
                title = f"Dataset Coverage Heatmap - Town{town:02d}"
            fig.suptitle(title, fontsize=12)

            out_path = out_dir / f"{args.output_prefix}_town{town:02d}.png"
            fig.tight_layout()
            fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            print(f"[DONE] Saved heatmap: {out_path}")
        finally:
            sensor.stop()
            sensor.destroy()


if __name__ == "__main__":
    main()
