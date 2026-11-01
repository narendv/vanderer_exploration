"""
Occupancy grid plus the NBVP tree, for eyeballing what the planner considered.

The tree is the useful part: if branches never leave the road, the collision
check is doing its job; if the best branch points back at mapped road, the
gain is not seeing what the camera would.
"""

import os

import numpy as np

from baselines.yamauchi.grid_map import FREE, OCC


def save_tree_png(path, grid, snapshot=None, tree=None, vehicle_xy=None, goal_xy=None, history=None):
    """Render map, tree, best branch and chosen edge. Cropped to the explored region."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if snapshot is None:
        snapshot = grid.snapshot()

    known = snapshot != -1
    if not np.any(known):
        return None

    ix, iy = np.nonzero(known)
    pad = int(20.0 / grid.res)
    x0, x1 = max(0, ix.min() - pad), min(grid.nx, ix.max() + pad + 1)
    y0, y1 = max(0, iy.min() - pad), min(grid.ny, iy.max() + pad + 1)
    sub = snapshot[x0:x1, y0:y1]

    # RGB: grey unknown, white free, black occupied.
    img = np.full(sub.shape + (3,), 128, dtype=np.uint8)
    img[sub == FREE] = (255, 255, 255)
    img[sub == OCC] = (0, 0, 0)
    # Transpose so world +x is right and +y is up in the saved image.
    img = np.transpose(img, (1, 0, 2))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    extent = [
        grid.origin_x + x0 * grid.res,
        grid.origin_x + x1 * grid.res,
        grid.origin_y + y0 * grid.res,
        grid.origin_y + y1 * grid.res,
    ]
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")

    if history:
        hx, hy = zip(*history)
        ax.plot(hx, hy, ".", ms=3, c="tab:purple", label="history (%d)" % len(history))
    if tree is not None:
        edges = tree.edges()
        if edges:
            ax.add_collection(LineCollection(edges, colors="tab:blue", linewidths=0.6, alpha=0.7))
        branch = tree.best_branch_nodes()
        if len(branch) > 1:
            ax.plot([n.x for n in branch], [n.y for n in branch], "-o", c="tab:orange", ms=3,
                    lw=1.5, label="best branch (gain %.0f m2)" % tree.best_gain)
    if vehicle_xy is not None:
        ax.scatter([vehicle_xy[0]], [vehicle_xy[1]], s=90, c="tab:green", marker="^", label="vehicle")
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=90, c="tab:red", marker="*", label="edge goal")

    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
