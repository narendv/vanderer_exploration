"""
Nearest-frontier selection and grid path planning.

Yamauchi's rule is "drive to the nearest frontier", where *nearest* means
nearest by traversable path, not by Euclidean distance. Both come out of a
single Dijkstra pass from the vehicle over the inflated free grid:

  * cost-to-go for every reachable cell, so the nearest frontier is one argmin
  * predecessors, so the path to it needs no second search
  * unreachable cells stay at infinity, which is what lets the purely
    geometric variant work without a road-network prior -- frontiers inside
    buildings or across walls are simply never reachable and drop out

Dijkstra runs through scipy.sparse.csgraph over a graph built only from
passable cells (a few 1e5 at most), not over the full town grid, so it stays
in the low milliseconds.
"""

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .grid_map import FREE, OCC

_OFFSETS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, np.sqrt(2.0)),
    (-1, 1, np.sqrt(2.0)),
    (1, -1, np.sqrt(2.0)),
    (1, 1, np.sqrt(2.0)),
]


class GridRoute(object):
    """Result of one Dijkstra pass: reachability plus path extraction."""

    def __init__(
        self, grid, passable, compact_id, dist, predecessors, start_cell, cell_ix, cell_iy
    ):
        self.grid = grid
        self.passable = passable
        self.compact_id = compact_id
        self.dist = dist
        self.predecessors = predecessors
        self.start_cell = start_cell
        # Compact-id -> cell lookup, in the same order compact_id was assigned.
        self.cell_ix = cell_ix
        self.cell_iy = cell_iy

    def cost_to(self, cell):
        cid = self.compact_id[cell[0], cell[1]]
        if cid < 0:
            return np.inf
        return float(self.dist[cid])

    def path_to(self, cell):
        """Cell path from start to `cell`, or [] if unreachable."""
        cid = self.compact_id[cell[0], cell[1]]
        if cid < 0 or not np.isfinite(self.dist[cid]):
            return []

        cells = []
        cur = cid
        while cur >= 0:
            cells.append((int(self.cell_ix[cur]), int(self.cell_iy[cur])))
            cur = self.predecessors[cur]
        cells.reverse()
        return cells


def _inflate(snapshot, grid, inflation_radius_m):
    """Grow obstacles by the vehicle's half-width so paths keep clearance."""
    occ = snapshot == OCC
    r_cells = int(np.ceil(inflation_radius_m / grid.res))
    if r_cells <= 0:
        return occ
    yy, xx = np.mgrid[-r_cells : r_cells + 1, -r_cells : r_cells + 1]
    disk = (xx ** 2 + yy ** 2) <= r_cells ** 2
    return ndimage.binary_dilation(occ, structure=disk)


def _nearest_passable(passable, cell, max_radius_cells):
    """Closest passable cell to `cell`, searching outward. None if none near."""
    if passable[cell[0], cell[1]]:
        return cell

    nx, ny = passable.shape
    for r in range(1, int(max_radius_cells) + 1):
        lo_x, hi_x = max(0, cell[0] - r), min(nx, cell[0] + r + 1)
        lo_y, hi_y = max(0, cell[1] - r), min(ny, cell[1] + r + 1)
        window = passable[lo_x:hi_x, lo_y:hi_y]
        if not np.any(window):
            continue
        wix, wiy = np.nonzero(window)
        gix, giy = wix + lo_x, wiy + lo_y
        d2 = (gix - cell[0]) ** 2 + (giy - cell[1]) ** 2
        best = int(np.argmin(d2))
        return (int(gix[best]), int(giy[best]))
    return None


def build_route(grid, snapshot, start_xy, inflation_radius_m=1.5, snap_radius_m=4.0):
    """One Dijkstra pass over the inflated free grid from `start_xy`."""
    blocked = _inflate(snapshot, grid, inflation_radius_m)
    passable = (snapshot == FREE) & (~blocked)
    if not np.any(passable):
        return None

    six, siy = grid.world_to_cell(start_xy[0], start_xy[1])
    six, siy = int(six), int(siy)
    if not grid.in_bounds(six, siy):
        return None

    snap_cells = int(np.ceil(snap_radius_m / grid.res))
    start_cell = _nearest_passable(passable, (six, siy), snap_cells)
    if start_cell is None:
        return None

    ix, iy = np.nonzero(passable)
    n = ix.size
    compact_id = np.full(passable.shape, -1, dtype=np.int64)
    compact_id[ix, iy] = np.arange(n)

    rows, cols, weights = [], [], []
    nx, ny = passable.shape
    for dx, dy, w in _OFFSETS:
        jx, jy = ix + dx, iy + dy
        ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
        # Index only the in-bounds entries; passable[jx, jy] on out-of-range
        # indices would wrap and silently create edges through walls.
        ok_idx = np.nonzero(ok)[0]
        if ok_idx.size == 0:
            continue
        neighbour_ok = passable[jx[ok_idx], jy[ok_idx]]
        sel = ok_idx[neighbour_ok]
        if sel.size == 0:
            continue
        rows.append(compact_id[ix[sel], iy[sel]])
        cols.append(compact_id[jx[sel], jy[sel]])
        weights.append(np.full(sel.size, w))

    if not rows:
        return None

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    weights = np.concatenate(weights)
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))

    src = int(compact_id[start_cell[0], start_cell[1]])
    dist, predecessors = dijkstra(
        graph, directed=False, indices=src, return_predecessors=True
    )
    return GridRoute(
        grid, passable, compact_id, dist, predecessors, start_cell, ix, iy
    )


def _ahead_fn(grid, vehicle_xy, vehicle_yaw_deg, cone_deg):
    """Mask function: cells within cone_deg of the heading, seen from the vehicle."""
    hx = np.cos(np.radians(vehicle_yaw_deg))
    hy = np.sin(np.radians(vehicle_yaw_deg))
    cos_cone = np.cos(np.radians(cone_deg))

    def ahead(ix, iy):
        wx, wy = grid.cell_to_world(ix, iy)
        dx, dy = wx - vehicle_xy[0], wy - vehicle_xy[1]
        return dx * hx + dy * hy >= cos_cone * np.hypot(dx, dy)

    return ahead


def select_nearest(
    route,
    clusters,
    grid,
    snap_radius_m=3.0,
    min_goal_distance_m=8.0,
    vehicle_xy=None,
    vehicle_yaw_deg=None,
    forward_cone_deg=0.0,
):
    """Reachable cluster with the lowest path cost. Returns (cluster, cells).

    A cluster's cost is the cost to its *closest member cell*, which is how
    explore_lite scores frontiers. Scoring by the centroid instead breaks on
    long arcs: the centroid of a frontier wrapping around the sensor's field
    of view sits next to the vehicle, so the greedy rule would keep picking a
    goal it is already standing on and never travel anywhere.

    `min_goal_distance_m` discards candidate cells closer than that. With a
    90-degree forward-facing camera the space beside and behind the vehicle is
    never observed, so free cells a metre away permanently border unknown
    space and are permanently the "nearest" frontier. Without this guard the
    greedy rule locks onto a goal it is already standing on and the vehicle
    never travels. Cost is measured in metres of path, not Euclidean distance.

    `forward_cone_deg` > 0 keeps only candidate cells within that angle of
    the vehicle's heading, seen from `vehicle_xy`. The same forward camera
    leaves the space behind the vehicle unexplored, so the nearest frontier
    is often behind it -- and reaching it means a U-turn most roads are too
    narrow for.
    """
    if route is None or not clusters:
        return None, []

    ahead = None
    if forward_cone_deg > 0 and vehicle_xy is not None and vehicle_yaw_deg is not None:
        ahead = _ahead_fn(grid, vehicle_xy, vehicle_yaw_deg, forward_cone_deg)

    cost_grid = np.full(route.passable.shape, np.inf, dtype=np.float64)
    cost_grid[route.cell_ix, route.cell_iy] = route.dist

    snap_cells = int(np.ceil(snap_radius_m / grid.res))
    min_cost = min_goal_distance_m / grid.res  # route costs are in cells
    best_cluster, best_cost, best_cell = None, np.inf, None

    for c in clusters:
        in_cone = ahead(c.cells_ix, c.cells_iy) if ahead is not None else None
        if in_cone is not None and not np.any(in_cone):
            continue
        costs = cost_grid[c.cells_ix, c.cells_iy]
        keep = costs >= min_cost
        if in_cone is not None:
            keep &= in_cone
        costs = np.where(keep, costs, np.inf)
        k = int(np.argmin(costs))
        cost = float(costs[k])
        target = (int(c.cells_ix[k]), int(c.cells_iy[k]))

        if not np.isfinite(cost):
            # Whole cluster sits inside an obstacle's inflation band. Try to
            # route beside it instead of discarding it outright -- frontiers
            # hugging a kerb are common and still worth reaching.
            near = _nearest_passable(route.passable, c.cell, snap_cells)
            if near is None:
                continue
            if ahead is not None and not ahead(np.array([near[0]]), np.array([near[1]]))[0]:
                continue
            cost = route.cost_to(near)
            target = near
            if not np.isfinite(cost) or cost < min_cost:
                continue

        if cost < best_cost:
            best_cluster, best_cost, best_cell = c, cost, target

    if best_cluster is None or not np.isfinite(best_cost):
        return None, []
    return best_cluster, route.path_to(best_cell)


def _line_of_sight(passable, a, b):
    """Supercover-free Bresenham check that every cell on a->b is passable."""
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if not passable[x, y]:
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def simplify(cells, passable):
    """Greedy line-of-sight shortcutting of a cell path."""
    if len(cells) < 3:
        return list(cells)
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_of_sight(passable, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def to_waypoints(grid, cells, spacing_m=2.0):
    """Cell path -> world (x, y) waypoints roughly `spacing_m` apart.

    Interpolates *along* each segment rather than just filtering the cell
    vertices. After line-of-sight shortcutting a straight 30m leg is only two
    cells, and handing the PID controller a single target 30m away starves it
    of steering signal.
    """
    if len(cells) < 2:
        return []

    xs, ys = grid.cell_to_world(
        np.array([c[0] for c in cells]), np.array([c[1] for c in cells])
    )

    pts = []
    for k in range(1, len(xs)):
        x0, y0 = float(xs[k - 1]), float(ys[k - 1])
        x1, y1 = float(xs[k]), float(ys[k])
        seg = np.hypot(x1 - x0, y1 - y0)
        if seg < 1e-9:
            continue
        n = max(1, int(np.ceil(seg / spacing_m)))
        for i in range(1, n + 1):
            t = float(i) / n
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))

    return pts
