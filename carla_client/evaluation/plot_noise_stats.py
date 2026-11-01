"""Plot noise statistics from noise_stats.json."""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "noise_stats.json"

    with open(path) as f:
        entries = json.load(f)

    indices = list(int(e) for e in entries.keys())
    mus   = [np.array(e["noise_mu"]).reshape(8, 2) for e in entries.values()]   # (T, 8, 2)
    vars_ = [np.array(e["noise_var"]).reshape(8, 2) for e in entries.values()]  # (T, 8, 2)

    mu_x  = np.stack([m[:, 0] for m in mus])    # (T, 8)
    mu_y  = np.stack([m[:, 1] for m in mus])    # (T, 8)
    var_x = np.stack([v[:, 0] for v in vars_])  # (T, 8)
    var_y = np.stack([v[:, 1] for v in vars_])  # (T, 8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), tight_layout=True)
    fig.suptitle("Noise Statistics")

    ax = axes[0, 0]
    steps = list(entries.keys())
    for i in range(mu_x.shape[1]):
        ax.plot(steps, mu_x[:, i], label=f"a{i}")
    ax.set_title("Mean – x dim")
    ax.set_xlabel("step")
    ax.legend(fontsize=7, ncol=4)

    ax = axes[0, 1]
    for i in range(mu_y.shape[1]):
        ax.plot(steps, mu_y[:, i], label=f"a{i}")
    ax.set_title("Mean – y dim")
    ax.set_xlabel("step")
    ax.legend(fontsize=7, ncol=4)

    ax = axes[1, 0]
    for i in range(var_x.shape[1]):
        ax.plot(steps, var_x[:, i], label=f"a{i}")
    ax.set_title("Var diagonal – x dim")
    ax.set_xlabel("step")
    ax.legend(fontsize=7, ncol=4)

    ax = axes[1, 1]
    for i in range(var_y.shape[1]):
        ax.plot(steps, var_y[:, i], label=f"a{i}")
    ax.set_title("Var diagonal – y dim")
    ax.set_xlabel("step")
    ax.legend(fontsize=7, ncol=4)

    plt.savefig("noise_stats.png", dpi=150)
    print("Saved noise_stats.png")
    plt.show()


if __name__ == "__main__":
    main()
