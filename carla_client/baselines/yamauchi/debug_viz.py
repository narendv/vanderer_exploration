"""
Grid/frontier/path dumps for eyeballing what the baseline actually sees.

The single most useful debugging artefact here is a PNG of the occupancy grid:
if the free region does not trace the road network, the depth backprojection
or the height band is wrong, and no amount of planner tuning will help.
"""

import os

import numpy as np

from .frontier import frontier_mask
from .grid_map import FREE, OCC


def save_grid_png(path, grid, snapshot=None, clusters=None, vehicle_xy=None, goal_xy=None):
    """Render the map as an image. Cropped to the explored region."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if snapshot is None:
        snapshot = grid.snapshot()

    known = snapshot != -1
    if not np.any(known):
        return None

    # Crop to what has been observed, with a margin, so early frames are not
    # a few pixels of road on a town-sized canvas.
    ix, iy = np.nonzero(known)
    pad = int(20.0 / grid.res)
    x0, x1 = max(0, ix.min() - pad), min(grid.nx, ix.max() + pad + 1)
    y0, y1 = max(0, iy.min() - pad), min(grid.ny, iy.max() + pad + 1)

    sub = snapshot[x0:x1, y0:y1]
    front = frontier_mask(snapshot)[x0:x1, y0:y1]

    # RGB: grey unknown, white free, black occupied, red frontier.
    img = np.zeros(sub.shape + (3,), dtype=np.uint8)
    img[...] = 128
    img[sub == FREE] = (255, 255, 255)
    img[sub == OCC] = (0, 0, 0)
    img[front] = (220, 40, 40)

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

    if clusters:
        cx = [c.world[0] for c in clusters]
        cy = [c.world[1] for c in clusters]
        ax.scatter(cx, cy, s=18, c="tab:blue", marker="o", label="frontier clusters")
    if vehicle_xy is not None:
        ax.scatter([vehicle_xy[0]], [vehicle_xy[1]], s=90, c="tab:green", marker="^", label="vehicle")
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=90, c="tab:orange", marker="*", label="goal")

    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
