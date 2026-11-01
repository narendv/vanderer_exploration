"""
Edge collision check -- the 2D stand-in for volumetric_mapping's
getLineStatusBoundingBox, which nbvplanner runs on every candidate edge
(rrt.cpp:341-345, and again on re-inserted branch nodes at 437-442).

Upstream sweeps the robot's bounding box along parent -> child, extended by
`system/bbx/overshoot` past the child, and accepts the edge only if every cell
it touches is known FREE. Unknown counts as blocked, because rrt.cpp tests for
== kFree. The tree therefore grows only through space the sensor has already
seen empty; it plans *to* viewpoints that look into the unknown, never
*through* the unknown.

On the BEV grid, sweeping a box becomes checking the centreline against
obstacles inflated by the vehicle's half-width plus margin. That is exactly
the passable mask the Yamauchi planner routes over (planner.build_route), so
both baselines drive under the same clearance rule.

Root clearance -- the one addition upstream does not need
---------------------------------------------------------
A forward-mounted camera never observes the vehicle's own footprint, the space
beside it, or the ~3 m ahead of it that is below the bottom of the image. At
the start of a run and after every collision respawn the root therefore sits
in UNKNOWN space, and every edge out of it would be rejected. Upstream solves
the same bootstrapping problem with a scripted initialization motion
(exploration.cpp: rotate in place, then back up -- "necessary that the known
free space allows the planning of initial paths"). A car cannot spin in place,
so instead cells within `root_clearance_m` of the root pass unless they are
actually OCC: the vehicle is physically standing in that space. It also covers
a root inside the inflation band of a kerb it has pulled up beside. The radius
matches the Yamauchi planner's start-cell snap (build_route snap_radius_m=4.0),
which exists for the same reason.

The clearance only lets an edge *leave* the blind zone. The node itself and
the overshoot past it must lie in known drivable space, so the tree never
places a viewpoint in space nobody has observed -- which is exactly where an
unseen wall beside the car would be.
"""

import numpy as np

from baselines.yamauchi.grid_map import FREE, OCC

# Shared with the Yamauchi planner on purpose: identical obstacle inflation is
# what makes the two baselines' collision rates comparable.
from baselines.yamauchi.planner import _inflate


class EdgeChecker(object):
    def __init__(
        self,
        grid,
        snapshot,
        root_xy,
        inflation_radius_m=1.5,
        overshoot_m=1.0,
        root_clearance_m=4.0,
    ):
        self.grid = grid
        blocked = _inflate(snapshot, grid, inflation_radius_m)
        self.passable = (snapshot == FREE) & ~blocked
        self.occ = snapshot == OCC
        self.root_x = float(root_xy[0])
        self.root_y = float(root_xy[1])
        self.overshoot_m = float(overshoot_m)
        self.root_clearance_m = float(root_clearance_m)
        # Half-cell steps cannot skip a cell along the line.
        self.step = 0.5 * grid.res

    def points_clear(self, xs, ys, may_relax=None):
        """True if every point is drivable.

        Points flagged in `may_relax` (all, if None) may instead rely on the
        root clearance.
        """
        ix, iy = self.grid.world_to_cell(xs, ys)
        if not np.all(self.grid.in_bounds(ix, iy)):
            return False
        ok = self.passable[ix, iy]
        if np.all(ok):
            return True
        relaxed = np.hypot(xs - self.root_x, ys - self.root_y) <= self.root_clearance_m
        if may_relax is not None:
            relaxed &= may_relax
        return bool(np.all(ok | (relaxed & ~self.occ[ix, iy])))

    def edge_clear(self, x0, y0, x1, y1):
        """True if the vehicle can drive (x0, y0) -> (x1, y1), plus overshoot."""
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        if length < 1e-9:
            return self.points_clear(np.array([x0]), np.array([y0]))
        reach = length + self.overshoot_m
        t = np.append(np.arange(0.0, reach, self.step), reach)
        return self.points_clear(
            x0 + t * dx / length, y0 + t * dy / length, may_relax=t < length
        )
