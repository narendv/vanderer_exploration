import argparse
import json
import csv
import os
import glob
import re

def compute_failure_metrics(run_dir):
    """Return (collision_count, total_steps, percentage_failure) for a run."""
    trajectory_path = os.path.join(run_dir, "trajectory.csv")
    metadata_path = os.path.join(run_dir, "metadata.json")

    collision_count = 0
    total_steps = 0

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        # Support both spellings to be resilient to typos in upstream writers.
        episode_starts = metadata.get("episode_starts")
        collision_count = max(0, len(episode_starts) - 1)
        total_steps = len(metadata.get("waypoint_generation_times", []))

    percentage_failure = (collision_count / total_steps * 100.0)
    return collision_count, total_steps, round(percentage_failure, 3)

def main():
    parser = argparse.ArgumentParser(description="Aggregate exploration metrics across modes and seeds.")
    parser.add_argument("town", type=str, help="Town number (e.g., 4, 5)")
    parser.add_argument("--input_dir", type=str, default="../outputs", help="Base directory containing mode subdirectories")
    parser.add_argument("--output_dir", type=str, default="../outputs/metrics", help="Output directory for CSV file")
    
    args = parser.parse_args()
    town = args.town

    
    modes = ["vanderer/random", "vanderer/select", "vanderer/guide", "yamauchi", "nbvp"]
    base_dir = os.path.abspath(args.input_dir)
    
    results = []
    
    for mode in modes:
        mode_dir = os.path.join(base_dir, mode)
        if not os.path.isdir(mode_dir):
            print(f"Warning: Directory for mode '{mode}' not found at {mode_dir}")
            continue
            
        # Match directories like recorded_trajs_town_{town}_seed_{seed}
        pattern = os.path.join(mode_dir, f"*town_{town}_seed_*", "exploration_metrics.json")
        json_files = glob.glob(pattern)
        
        for json_file in json_files:
            # Extract seed from folder name
            folder_name = os.path.basename(os.path.dirname(json_file))
            match = re.search(r"seed_(\d+)", folder_name)
            seed = match.group(1) if match else "unknown"
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    aggregate = data.get("aggregate", {})
                    
                    total_path_length = round(aggregate.get("path_length_m", 0.0), 2)
                    total_area = aggregate.get("new_area_m2", 0.0)
                    area_per_meter = round(aggregate.get("area_per_meter", 0.0), 3)
                    
                    run_dir = os.path.dirname(json_file)
                    total_collisions, total_steps, percentage_failure = compute_failure_metrics(run_dir)
                    
                    results.append({
                        "town": town,
                        "seed": seed,
                        "mode": mode,
                        "total_path_length": total_path_length,
                        "total_area": total_area,
                        "area_per_path_length": area_per_meter,
                        "total_collisions": total_collisions,
                        "total_steps": total_steps,
                        "percentage_failure": percentage_failure,
                    })
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                
    if not results:
        print("No metrics found for the given town.")
        return

    # Sort results by mode and seed for better readability
    results.sort(key=lambda x: (x["mode"], int(x["seed"]) if x["seed"].isdigit() else x["seed"]))

    # Write to CSV
    csv_file = os.path.join(args.output_dir, f"aggregate_metrics_town_{town}.csv")
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            "town",
            "seed",
            "mode",
            "total_path_length",
            "total_area",
            "area_per_path_length",
            "total_collisions",
            "total_steps",
            "percentage_failure",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"Successfully aggregated {len(results)} records into {csv_file}")

if __name__ == "__main__":
    main()
