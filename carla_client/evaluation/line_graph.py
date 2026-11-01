import argparse
import os
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\boldmath',
    'font.size': 42,
    'font.weight': 'bold',
    'axes.titlesize': 42,
    'axes.titleweight': 'bold',
    'axes.labelsize': 42,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 40,
    'ytick.labelsize': 40,
    'legend.fontsize': 40,
})

from evaluation.plot_bev_trajectory import load_rows, load_metadata
from evaluation.plot_graph import compute_road_curves

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="../outputs/vanderer", help="Base directory of the experiments.")
    parser.add_argument("--output", type=str, default="line_graph.png")
    args = parser.parse_args()
    
    modes = ["random", "select", "guide"]
    labels = {
        "random": r"\textbf{Base Policy}",
        "select": r"\textbf{Greedy Selection}",
        "guide": r"\textbf{VANDERER}"
    }
    
    colors = {
        "random": "#2166AC",  # Blue
        "select": "#1B9E77",  # Green
        "guide": "#FF7F00"    # Orange
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    # Hardcoded pairs: edit these two entries to pick the runs shown.
    town_seeds = [(3, 3), (4, 3)]
    titles = [r"\textbf{(a) Example 1}", r"\textbf{(b) Example 2}"]
    
    lines_handles = []
    lines_labels = []

    for i, (town, seed) in enumerate(town_seeds):
        ax = axes[i]
        
        for mode in modes:
            folder_name = f"recorded_trajs_town_{town}_seed_{seed}"
            exp_dir = os.path.join(args.base_dir, mode, folder_name)
            
            if not os.path.exists(exp_dir):
                print(f"Path not found: {exp_dir}")
                continue
                
            metadata = load_metadata(exp_dir, "metadata.json")
            rows_by_ep = load_rows(exp_dir, "trajectory.csv")
            # Road coverage against steps, same definition as road_exploration_metrics.py.
            steps, coverage, _efficiency = compute_road_curves(rows_by_ep, metadata["world_name"])
            
            if not steps:
                print(f"No data for {mode} on town {town} seed {seed}")
                continue
                
            steps_scaled = [s / 1000.0 for s in steps]
            
            line, = ax.plot(steps_scaled, coverage, color=colors[mode], label=labels[mode], linewidth=3.0)
            
            if i == 0:
                lines_handles.append(line)
                lines_labels.append(labels[mode])
                
        ax.set_xlabel(r"\textbf{Steps ($\times 10^3$)}" + "\n" + titles[i], fontsize=32)
        ax.set_ylabel(r"\textbf{Road Coverage} (\textbf{\%})", fontsize=32)
        
        ax.grid(True, linewidth=2.0)
        
        # Axis styling
        ax.tick_params(axis="both", which="major", width=3, length=8, labelsize=30)
        for label_obj in ax.get_xticklabels() + ax.get_yticklabels():
            label_obj.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(4.0)
            
    # Add a unified legend at the top
    # We create the legend on the figure
    if lines_handles:
        fig.legend(lines_handles, lines_labels, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.87),
                   ncol=3, fancybox=False, edgecolor="black", framealpha=1.0,
                   fontsize=30, prop={'size': 30})
    
    plt.tight_layout()
    # Add extra space at top for legend
    plt.subplots_adjust(top=0.85)
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved figure to {args.output}")

if __name__ == "__main__":
    main()
