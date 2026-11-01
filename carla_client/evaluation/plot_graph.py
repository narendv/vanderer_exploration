import argparse
import os
import math
from collections import defaultdict
from typing import Dict, List

import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import carla

# Apply science style and set base font size
plt.style.use(["science"]) #, "grid"
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\boldmath',
    'font.size': 26,
    'font.weight': 'bold',
    'axes.titlesize': 26,
    'axes.titleweight': 'bold',
    'axes.labelsize': 26,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
})


from evaluation.road_exploration_metrics import town_model, DRIVING  # offline road network (no server)
from evaluation.plot_bev_trajectory import (
    load_metadata,
    load_rows,
    build_camera,
    project_world_to_pixel,
    capture_topdown,
    _sample_segment_2d,
)


def compute_road_curves(
    rows_by_episode: Dict[int, List[Dict[str, str]]],
    world_name: str,
    step_every: int = 500,
    piece_m: float = 2.0,
    sample_m: float = 0.5,
    max_jump_m: float = 2.0,
):
    """Road coverage (%) and road exploration efficiency against step count.

    Same definitions as road_exploration_metrics.py: the town's drivable network
    is split into piece_m segments (each junction is one segment), the path is
    sampled every sample_m metres and each sample marks the segment beneath it.
    Coverage is visited length over the town's total; efficiency is visited
    length over distance driven. A move longer than max_jump_m between
    consecutive rows is a teleport: not driven distance, and not sampled.
    """
    town, town_pieces, town_junctions = town_model(world_name, piece_m)
    total_road = len(town_pieces) * piece_m + sum(town_junctions.values())

    pieces, junctions = set(), set()

    def visit(x, y, z):
        wp = town.get_waypoint(carla.Location(x=x, y=y, z=z),
                               project_to_road=False, lane_type=DRIVING)
        if wp is None:
            return
        if wp.is_junction:
            junctions.add(wp.junction_id)
        else:
            pieces.add((wp.road_id, int(wp.s // piece_m)))

    def explored_m():
        return (len(pieces & town_pieces) * piece_m
                + sum(town_junctions[j] for j in junctions & set(town_junctions)))

    steps, coverage, efficiency = [], [], []
    path_m = 0.0
    carry = 0.0
    global_step = 0
    next_mark = step_every

    for ep in sorted(rows_by_episode):
        prev = None
        for r in rows_by_episode[ep]:
            x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
            if prev is None:
                visit(x, y, z)
            else:
                global_step += 1
                seg = math.hypot(x - prev[0], y - prev[1])
                if seg > max_jump_m:              # teleport: skip it entirely
                    carry = 0.0
                else:
                    path_m += seg
                    d = sample_m - carry
                    while d <= seg:
                        t = d / seg
                        visit(prev[0] + (x - prev[0]) * t,
                              prev[1] + (y - prev[1]) * t,
                              prev[2] + (z - prev[2]) * t)
                        d += sample_m
                    carry = (seg - (d - sample_m)) if seg > 0 else carry
                if global_step >= next_mark:
                    road = explored_m()
                    steps.append(global_step)
                    coverage.append(100.0 * road / total_road if total_road else 0.0)
                    efficiency.append(road / path_m if path_m > 1e-9 else 0.0)
                    next_mark += step_every
            prev = (x, y, z)
    return steps, coverage, efficiency


def compute_cumulative_metrics(
    rows_by_episode: Dict[int, List[Dict[str, str]]],
    cell_size_m: float = 4.0,
    interp_step_m: float = 0.5,
):
    """
    Returns step indices and cumulative (area_m2 / path_m) at each step.
    A step here means a point in the overall trajectory sequentially.
    """
    unique_cells = set()
    path_length_m = 0.0

    steps_recorded = []
    ratio_recorded = []
    
    global_step = 0
    
    # We will iterate episodes in order
    episodes = sorted(rows_by_episode.keys())
    for ep in episodes:
        rows = rows_by_episode[ep]
        if not rows:
            continue
            
        xy = [(float(r["x"]), float(r["y"])) for r in rows]
        
        # Add the first point's cell
        unique_cells.add(
            (int(math.floor(xy[0][0] / cell_size_m)), int(math.floor(xy[0][1] / cell_size_m)))
        )
        
        for idx in range(1, len(xy)):
            global_step += 1
            x0, y0 = xy[idx - 1]
            x1, y1 = xy[idx]
            
            dist = math.hypot(x1 - x0, y1 - y0)
            path_length_m += dist
            
            for sx, sy in _sample_segment_2d((x0, y0), (x1, y1), interp_step_m):
                cell = (int(math.floor(sx / cell_size_m)), int(math.floor(sy / cell_size_m)))
                unique_cells.add(cell)
            
            new_area_m2 = len(unique_cells) * (cell_size_m ** 2)
            ratio = new_area_m2 / path_length_m if path_length_m > 1e-9 else 0.0
            
            steps_recorded.append(global_step)
            ratio_recorded.append(ratio)

    # subsample to reduce number of points for plotting (e.g., every 100 steps)
    subsample_factor = 500
    steps_recorded = steps_recorded[1000::subsample_factor]
    ratio_recorded = ratio_recorded[1000::subsample_factor]

    return steps_recorded, ratio_recorded


def get_points_up_to_step(
    rows_by_episode: Dict[int, List[Dict[str, str]]],
    max_global_steps: int
):
    """
    Extract points exactly up to the `max_global_steps`.
    Returns list of (x, y, z, ep).
    """
    points = []
    global_step = 0
    episodes = sorted(rows_by_episode.keys())
    for ep in episodes:
        rows = rows_by_episode[ep]
        for idx, r in enumerate(rows):
            if idx > 0:
                global_step += 1
            if global_step > max_global_steps:
                return points
            points.append((float(r["x"]), float(r["y"]), float(r["z"]), ep))
    return points


def plot_progress_figure(
    points_dict,
    world,
    rgb,
    world_to_camera,
    k,
    image_size,
    output_path,
    title="Top-Down Auto"
):
    num_modes = len(points_dict)
    fig, axes = plt.subplots(1, num_modes, figsize=(10 * num_modes, 10), dpi=150)
    
    # Ensure axes is iterable even if there's only 1 mode
    if num_modes == 1:
        axes = [axes]
        
    # Pick fixed colors for different modes
    mode_colors = {
        "random": "blue",
        "select": "green",
        "guide": "red"
    }
    fallback_colors = ["purple", "orange", "cyan", "magenta", "yellow"]
    
    for idx, (ax, (mode, points)) in enumerate(zip(axes, points_dict.items())):
        ax.imshow(rgb)
        ax.set_title(f"Trajectory: {mode.capitalize()} (Step {title})", fontweight="bold")
        
        uv_by_ep = defaultdict(list)
        for x, y, z, ep in points:
            uv = project_world_to_pixel(
                carla.Location(x=float(x), y=float(y), z=float(z) + 0.3),
                world_to_camera,
                k,
            )
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < image_size and 0 <= v < image_size:
                uv_by_ep[ep].append((u, v))

        c = mode_colors.get(mode, fallback_colors[idx % len(fallback_colors)])
        
        for ep in sorted(uv_by_ep.keys()):
            uv = uv_by_ep[ep]
            if len(uv) < 2:
                continue
            u = [p[0] for p in uv]
            v = [p[1] for p in uv]
            
            # Using alpha=0.8 so multiple episodes overlay gently
            ax.plot(u, v, color=c, linewidth=5.0, alpha=0.8, zorder=2)
            ax.scatter(u[0], v[0], color=c, marker="o", s=30, zorder=3)
            ax.scatter(u[-1], v[-1], color=c, marker="x", s=30, zorder=3)
            
        ax.set_axis_off()
        
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Plot metrics and BEV trajectory periodically")
    parser.add_argument("--town", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--interval_steps", type=int, default=20000)
    parser.add_argument("--modes", nargs="+", default=["random", "select", "guide"])
    parser.add_argument("--base_dir", type=str, default="../outputs/vanderer")
    parser.add_argument("--curve_every_steps", type=int, default=500,
                        help="Sample the road-metric curves this often, in steps.")
    parser.add_argument("--apl", action="store_true",
                        help="Also emit the old grid A/PL graph as metrics.png.")
    args = parser.parse_args()

    
    save_base_dir = os.path.join(args.base_dir, "plots", f"town_{args.town}_seed_{args.seed}")
    os.makedirs(save_base_dir, exist_ok=True)
    # We will need just one connection to carla to get the background image 
    # to avoid reloading map and waiting multiple times.
    modes_data = {}
    
    for mode in args.modes:
        folder_name = f"recorded_trajs_town_{args.town}_seed_{args.seed}"
        exp_dir = os.path.join(args.base_dir, mode, folder_name)
            
        metadata = load_metadata(exp_dir, "metadata.json")
        rows_by_ep = load_rows(exp_dir, "trajectory.csv")
        modes_data[mode] = {
            "metadata": metadata,
            "rows_by_ep": rows_by_ep
        }

    if not modes_data:
        print("No valid data found for the given town/seed across modes.")
        return

    # 1. Compute and plot the road-network curves (one figure per metric)
    max_steps_overall = 0
    curves = {}
    for mode, data in modes_data.items():
        print(f"Computing road metrics for {mode}...")
        steps, coverage, efficiency = compute_road_curves(
            data["rows_by_ep"], data["metadata"]["world_name"], args.curve_every_steps)
        curves[mode] = (steps, coverage, efficiency)
        if steps:
            max_steps_overall = max(max_steps_overall, steps[-1])

    def style_axes(ax):
        ax.tick_params(axis="both", which="major", labelsize=24, width=3, length=8)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(4.0)
        ax.grid(True, linewidth=2.0)

    for idx, (key, ylabel, fname) in enumerate(
            [(1, r"\textbf{Road Coverage} (\textbf{\%})", "metrics_rc.png"),
             (2, r"\textbf{Road Expl. Efficiency}", "metrics_ree.png")]):
        plt.figure(figsize=(10, 6), dpi=150)
        for mode in modes_data:
            steps, coverage, efficiency = curves[mode]
            series = coverage if key == 1 else efficiency
            plt.plot([s / 1000.0 for s in steps], series, label=mode, linewidth=3.0)
        plt.xlabel(r"\textbf{Steps ($\times 10^3$)}", fontsize=28)
        plt.ylabel(ylabel, fontsize=28)
        style_axes(plt.gca())
        out = os.path.join(save_base_dir, fname)
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"{'Road coverage' if key == 1 else 'Road efficiency'} graph saved to {out}")

    if args.apl:   # the old grid A/PL curve, off by default
        plt.figure(figsize=(10, 6), dpi=150)
        for mode, data in modes_data.items():
            steps, ratios = compute_cumulative_metrics(data["rows_by_ep"])
            plt.plot([s / 1000.0 for s in steps], ratios, label=mode, linewidth=3.0)
        plt.xlabel(r"\textbf{Steps ($\times 10^3$)}", fontsize=28)
        plt.ylabel(r"\textbf{A/PL} ($\mathbf{m}$)", fontsize=28)
        style_axes(plt.gca())
        graph_out = os.path.join(save_base_dir, "metrics.png")
        plt.savefig(graph_out, bbox_inches="tight")
        plt.close()
        print(f"A/PL graph saved to {graph_out}")
    
    # 2. Plot Progress Figures every N steps
    # Connect to Carla
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(40.0)
    
    # Pick world name from the first valid metadata
    first_mode = list(modes_data.keys())[0]
    wanted_world = modes_data[first_mode]["metadata"].get("world_name")
    world = client.load_world(wanted_world) if wanted_world else client.get_world()
    print(f"Loaded / Verified world: {world.get_map().name}")

    # Calculate camera transform based on all points from all modes to get a consistent unified view
    all_points = []
    for mode, data in modes_data.items():
        for ep in data["rows_by_ep"].values():
            for r in ep:
                all_points.append((float(r["x"]), float(r["y"]), float(r["z"])))
                
    if not all_points:
        print("No points available to compute camera position.")
        return

    xs = np.array([p[0] for p in all_points])
    ys = np.array([p[1] for p in all_points])
    zs = np.array([p[2] for p in all_points])
    
    cx = (xs.max() + xs.min()) / 2.0
    cy = (ys.max() + ys.min()) / 2.0
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) + 60.0
    z_base = float(np.percentile(zs, 95))
    height = max(40.0, span / (2.0 * np.tan(np.radians(args.fov_deg) / 2.0)))
    
    cam_tf = carla.Transform(
        carla.Location(x=cx, y=cy, z=z_base + height),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )

    print("Capturing background RGB...")
    rgb, sensor = capture_topdown(world, cam_tf, args.image_size, args.fov_deg)
    
    try:
        world_to_camera = np.array(sensor.get_transform().get_inverse_matrix(), dtype=np.float64)
        k = build_camera(args.image_size, args.image_size, args.fov_deg)
        
        # for each multiple of args.interval_steps, collect all modes
        current_interval = args.interval_steps
        while current_interval <= max_steps_overall:
            points_dict = {}
            for mode, data in modes_data.items():
                points_up_to = get_points_up_to_step(data["rows_by_ep"], current_interval)
                if points_up_to:
                    points_dict[mode] = points_up_to
            
            if points_dict:
                out_path = os.path.join(
                    save_base_dir,
                    f"progress_combined_step_{current_interval}.png"
                )
                plot_progress_figure(
                    points_dict,
                    world,
                    rgb,
                    world_to_camera,
                    k,
                    args.image_size,
                    out_path,
                    title=str(current_interval)
                )
                print(f"[Combined] Saved progress figure for step {current_interval} to {out_path}")
            
            current_interval += args.interval_steps

    finally:
        sensor.stop()
        sensor.destroy()


if __name__ == "__main__":
    main()
