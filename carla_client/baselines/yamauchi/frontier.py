"""
Frontier detection over the BEV occupancy grid.

A frontier cell is a known-free cell adjacent to unknown space -- Yamauchi's
original definition (Yamauchi 1997, "A Frontier-Based Approach for Autonomous
Exploration"). Cells are grouped into connected components and each component
above a minimum size becomes one candidate goal.

Clustering uses scipy.ndimage.label rather than a density-based clusterer:
that is what explore_lite and the wavefront-frontier-detector implementations
do, and it avoids a dependency on hdbscan.
"""

import numpy as np
from scipy import ndimage

from .grid_map import FREE, UNKNOWN


class FrontierCluster(object):
    __slots__ = ("cell", "world", "size", "cells_ix", "cells_iy")

    def __init__(self, cell, world, size, cells_ix, cells_iy):
        self.cell = cell  # (ix, iy) representative cell
        self.world = world  # (x, y) in CARLA world coords
        self.size = size  # number of frontier cells in the cluster
        # Every member cell. Selection needs these: the cost of a cluster is
        # the cost to its *closest* cell (explore_lite's definition), not to
        # its centroid. On a long frontier arc the centroid can sit right
        # beside the vehicle while the useful unexplored edge is far away.
        self.cells_ix = cells_ix
        self.cells_iy = cells_iy

    def __repr__(self):
        return "FrontierCluster(cell=%s, world=(%.1f, %.1f), size=%d)" % (
            self.cell,
            self.world[0],
            self.world[1],
            self.size,
        )


def frontier_mask(snapshot):
    """Known-free cells with at least one 4-connected unknown neighbour."""
    free = snapshot == FREE
    unknown = snapshot == UNKNOWN

    neigh = np.zeros_like(unknown)
    neigh[:-1, :] |= unknown[1:, :]
    neigh[1:, :] |= unknown[:-1, :]
    neigh[:, :-1] |= unknown[:, 1:]
    neigh[:, 1:] |= unknown[:, :-1]

    return free & neigh


def detect(grid, snapshot=None, min_frontier_cells=4):
    """Cluster the frontier mask into candidate goals, largest first."""
    if snapshot is None:
        snapshot = grid.snapshot()

    mask = frontier_mask(snapshot)
    if not np.any(mask):
        return []

    structure = np.ones((3, 3), dtype=np.int8)  # 8-connectivity
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return []

    clusters = []
    objects = ndimage.find_objects(labels)
    for label_id in range(1, n + 1):
        sl = objects[label_id - 1]
        if sl is None:
            continue
        sub = labels[sl] == label_id
        size = int(np.count_nonzero(sub))
        if size < min_frontier_cells:
            continue

        local_ix, local_iy = np.nonzero(sub)
        ix = local_ix + sl[0].start
        iy = local_iy + sl[1].start

        # Snap the centroid to an actual member cell: on curved or L-shaped
        # fronts the raw centroid can land on a non-frontier (or occupied) cell.
        cx, cy = ix.mean(), iy.mean()
        best = np.argmin((ix - cx) ** 2 + (iy - cy) ** 2)
        rep = (int(ix[best]), int(iy[best]))
        wx, wy = grid.cell_to_world(rep[0], rep[1])
        clusters.append(
            FrontierCluster(rep, (float(wx), float(wy)), size, ix, iy)
        )

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


class Blacklist(object):
    """Suppress goals we have already failed to reach.

    explore_lite keeps the same structure. Without it the greedy
    nearest-frontier rule oscillates: the vehicle abandons a goal, immediately
    re-detects the same frontier as nearest, and re-targets it forever.
    """

    def __init__(self, radius_m=6.0):
        self.radius_m = float(radius_m)
        self.points = []

    def add(self, world_xy):
        self.points.append((float(world_xy[0]), float(world_xy[1])))

    def contains(self, world_xy):
        if not self.points:
            return False
        px = np.array([p[0] for p in self.points])
        py = np.array([p[1] for p in self.points])
        d2 = (px - world_xy[0]) ** 2 + (py - world_xy[1]) ** 2
        return bool(np.any(d2 <= self.radius_m ** 2))

    def filter(self, clusters):
        return [c for c in clusters if not self.contains(c.world)]

    def __len__(self):
        return len(self.points)
