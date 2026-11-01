"""Summary bar figure: road coverage, exploration efficiency and policy failure.

Three panels side by side, one metric each, with one bar per method pooled over
every complete run (all towns, all seeds). The metrics differ by two orders of
magnitude and two of them point in opposite directions, so each gets its own
axis rather than sharing one.

Error bars default to the standard error of the mean: with a single pooled bar
the question is how well the mean is pinned down, not how far individual runs
scatter. Use --err sd for the run-to-run spread instead.

Usage:
  python -m evaluation.coverage_pf_bar --csv ../outputs/metrics/road_metrics_100k.csv
                            [--out coverage_pf_bar.png] [--err sem|sd]
"""

import argparse
import csv
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the "science" style)

# Categorical slots 1-3 of the validated palette, in fixed order.
# (CSV method key, display name, colour). Display names are the paper's terms;
# the keys still match road_exploration_metrics.py's output.
METHODS = [("random", "Base\nPolicy", "#2a78d6"),
           ("select", "Greedy\nSelection", "#eb6834"),
           ("guide", "VANDERER\n(ours)", "#1baf7a")]   # real newlines: matplotlib typesets each line
# key, panel label, direction note, value format, ceiling (None = unbounded)
# Every metric here is always written to the companion CSV; PANELS selects which
# ones are drawn, so commenting a row out drops a panel without losing the data.
METRICS = [("cov", r"Road Coverage (\%)", "higher is better", "%.1f", 100.0),
           ("eff", r"Exploration Efficiency", "higher is better", "%.3f", 1.0),
           ("pf", r"Policy Failure (\%)", "lower is better", "%.2f", None)]
PANELS = [m for m in METRICS if m[0] in (
    "cov",
    "eff",
    "pf",
)]
SURFACE = "#f7f7f7"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.style.use(["science"])
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\boldmath",
    "font.size": 16, "font.weight": "bold",
    "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 16, "axes.labelweight": "bold",
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
})

bold = lambda s: r"\textbf{" + s + "}"


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default=os.path.join(here, "coverage_pf_bar.png"))
    p.add_argument("--err", choices=["sem", "sd"], default="sem")
    p.add_argument("--seeds", default="1,2,3",
                   help="Seeds to pool, comma-separated ('all' for every seed). The default keeps "
                        "the matched replan-40 set: random's town-2 seeds 41-43 were re-run at "
                        "replan 20 and would otherwise mix two configurations into one bar.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load(path, seeds=None):
    """{method: {metric: [per-run values]}} over complete runs of the chosen seeds."""
    runs = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["complete"] != "True":
            continue
        if seeds is not None and int(r["seed"]) not in seeds:
            continue
        calls = float(r["policy_calls"] or 0) or float(r["rows_used"]) / 40.0
        runs[r["method"]]["cov"].append(100.0 * float(r["road_coverage"]))
        runs[r["method"]]["eff"].append(float(r["road_efficiency"]))
        runs[r["method"]]["pf"].append(100.0 * float(r["collisions"]) / calls)
    return runs


def main():
    args = parse_args()
    seeds = None if args.seeds.strip().lower() == "all" else {int(x) for x in args.seeds.split(",")}
    runs = load(args.csv, seeds)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.7 * len(PANELS), 4.3), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("white")
    x = np.arange(len(METHODS))
    summary = {}

    for key, ylabel, note, fmt, ceiling in METRICS:   # stats for every metric, drawn only if in PANELS
        mu = np.array([st.mean(runs[m][key]) for m, _, _ in METHODS])
        sd = np.array([st.stdev(runs[m][key]) for m, _, _ in METHODS])
        n = np.array([len(runs[m][key]) for m, _, _ in METHODS])
        err = sd if args.err == "sd" else sd / np.sqrt(n)
        # Both measures are bounded; a whisker past the bound would imply values
        # the metric cannot take.
        lo = np.minimum(err, mu)
        hi = err if ceiling is None else np.minimum(err, ceiling - mu)
        summary[key] = (mu, sd, err, n)
        if (key, ylabel, note, fmt, ceiling) not in PANELS:
            continue
        ax = axes[PANELS.index((key, ylabel, note, fmt, ceiling))]

        for i, (m, label, color) in enumerate(METHODS):
            ax.bar(x[i], mu[i], 0.62, color=color, zorder=3,
                   yerr=[[lo[i]], [hi[i]]],
                   error_kw=dict(ecolor=MUTED, elinewidth=1.3, capsize=3.5, zorder=4))
            ax.text(x[i], mu[i] + hi[i], bold(fmt % mu[i]), ha="center", va="bottom",
                    fontsize=12, color=INK, zorder=5)
        ax.set_ylabel(bold(ylabel), labelpad=6)
        ax.set_facecolor(SURFACE)
        ax.set_xticks(x)
        # Method names on the axis, so identity never rests on colour alone.
        ax.set_xticklabels(["\n".join(bold(line) for line in label.split("\n"))
                            for _, label, _ in METHODS])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, colors=INK)
        ax.set_ylim(0, None)
        ax.margins(y=0.22)
        ax.set_title(bold(note), fontsize=12, color=MUTED, pad=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print("figure: %s  (error bars: %s, seeds: %s)" % (os.path.abspath(args.out), args.err, args.seeds))

    # Table view: companion to the figure, and the relief the contrast check asks for.
    table = os.path.splitext(args.out)[0] + "_data.csv"
    with open(table, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "mean", "sd", "sem", "n_runs"])
        for key, _, _, _, _ in METRICS:
            mu, sd, err, n = summary[key]
            for i, (m, _, _) in enumerate(METHODS):
                w.writerow([m, key, round(mu[i], 4), round(sd[i], 4), round(sd[i] / np.sqrt(n[i]), 4), n[i]])
    print("table:  %s" % os.path.abspath(table))
    print()
    print("%-8s %22s %22s %22s" % ("method", "coverage % (sd|sem)", "efficiency", "PF %"))
    for i, (m, _, _) in enumerate(METHODS):
        cells = []
        for key, _, _, _, _ in METRICS:
            mu, sd, err, n = summary[key]
            cells.append("%8.3f %5.3f|%5.3f n=%d" % (mu[i], sd[i], sd[i] / np.sqrt(n[i]), n[i]))
        print("%-8s %s" % (m, " ".join(cells)))


if __name__ == "__main__":
    main()
