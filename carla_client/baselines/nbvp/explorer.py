"""
NBVPExplorer -- the whole NBVP baseline behind one object, with the same
integrate() / replan() / abandon_goal() surface as YamauchiExplorer.

Mapping is YamauchiExplorer's, unchanged: the same GridMap, integrated from the
same depth frames with the same parameters. The two baselines differ only in
how they pick where to drive.

replan() is nbvplanner's plannerCallback (nbvp.hpp:184-207):

    clear the tree; re-insert the remembered best branch
    while no gain found, or fewer than initial_iterations nodes:
        if more than cutoff_iterations nodes:   no gain anywhere
        if more than 1000 * (nodes + 1) attempts: sampling has stalled
        grow the tree by one sampling attempt
    return the first edge of the best branch
    remember the rest of that branch; push the root onto the history

Receding horizon: only the first edge is driven, then the controller calls
replan() again from wherever the vehicle ended up.

Option (a): no-gain handling
----------------------------
Upstream answers "no gain anywhere" with ros::shutdown(), and "sampling has
stalled" with getPathBackToPrevious(), a straight path back to the root of the
last executed edge. Here both backtrack. Only when the history is empty and
there is still no gain does replan() report `finished` -- the point where
upstream would have shut down with nothing left to try. Backtracking walks the
vehicle back along the edges it drove, replanning from each earlier pose, so
it can recover at a junction whose other branch it has not looked down yet.

Stalled sampling with gain in hand
----------------------------------
Upstream also backtracks when sampling stalls before initial_iterations nodes,
even if the tree already sees unmapped space. In CARLA that happens whenever
the mapped road ahead only fits a few 8 m edges -- at spawn, and after every
respawn. It stranded the vehicle at spawn (Town01 seed 2: 50 stalled replans,
each already seeing 50-145 m^2 of unknown), and the resulting straight-line
backtracks are U-turns (11 of seed 1's 35 collisions). Here a stalled tree
that has found gain drives its best edge (reason `ok_capped`); only a tree
with no gain backtracks.
"""

import time

import numpy as np

from .collision import EdgeChecker
from .tree import RrtTree
from .views import ViewModel


class ReplanResult(object):
    __slots__ = (
        "waypoints",
        "goal",
        "reason",
        "backtrack",
        "n_nodes",
        "n_attempts",
        "best_gain",
        "latency_s",
    )

    def __init__(
        self,
        waypoints=None,
        goal=None,
        reason="ok",
        backtrack=False,
        n_nodes=0,
        n_attempts=0,
        best_gain=0.0,
        latency_s=0.0,
    ):
        self.waypoints = waypoints or []
        self.goal = goal
        self.reason = reason
        self.backtrack = backtrack
        self.n_nodes = n_nodes
        self.n_attempts = n_attempts
        self.best_gain = best_gain
        self.latency_s = latency_s

    def __repr__(self):
        return "ReplanResult(%s, n_wp=%d, nodes=%d, gain=%.1f)" % (
            self.reason,
            len(self.waypoints),
            self.n_nodes,
            self.best_gain,
        )


def segment_waypoints(a, b, spacing_m=2.0):
    """Points roughly `spacing_m` apart along a -> b, ending at b, excluding a.

    Upstream's samplePath does the same interpolation at v_max * dt; the PID
    wants targets a couple of metres apart, as in the Yamauchi baseline.
    """
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if length < 1e-6:
        return []
    n = max(1, int(np.ceil(length / spacing_m)))
    return [
        (a[0] + (b[0] - a[0]) * i / float(n), a[1] + (b[1] - a[1]) * i / float(n))
        for i in range(1, n + 1)
    ]


class NBVPExplorer(object):
    def __init__(
        self,
        grid,
        hfov_deg=90.0,
        vfov_deg=73.74,
        cam_height_m=1.7,
        cam_forward_m=1.0,
        max_range_m=40.0,
        min_range_m=2.0,
        obstacle_band_m=(0.3, 3.0),
        pixel_stride=2,
        extension_range_m=8.0,
        gain_range_m=12.0,
        degressive_coeff=0.0625,
        zero_gain=0.0,
        initial_iterations=15,
        cutoff_iterations=200,
        inflation_radius_m=1.5,
        overshoot_m=1.0,
        root_clearance_m=4.0,
        forward_cone_deg=0.0,
        waypoint_spacing_m=2.0,
        seed=0,
    ):
        self.grid = grid
        self.view = ViewModel(
            grid.res,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            gain_range_m=gain_range_m,
            cam_height_m=cam_height_m,
            cam_forward_m=cam_forward_m,
        )
        self.rng = np.random.default_rng(seed)

        self.hfov_deg = hfov_deg
        self.max_range_m = max_range_m
        self.min_range_m = min_range_m
        self.obstacle_band_m = obstacle_band_m
        self.pixel_stride = pixel_stride

        self.extension_range_m = extension_range_m
        self.degressive_coeff = degressive_coeff
        self.zero_gain = zero_gain
        self.initial_iterations = initial_iterations
        self.cutoff_iterations = cutoff_iterations
        self.inflation_radius_m = inflation_radius_m
        self.overshoot_m = overshoot_m
        self.root_clearance_m = root_clearance_m
        self.forward_cone_deg = forward_cone_deg
        self.waypoint_spacing_m = waypoint_spacing_m

        # Upstream history_: roots of the edges driven so far, newest last.
        self.history = []
        # Upstream bestBranchMemory_, shallowest node first.
        self.branch_memory = []
        self.current_goal = None
        self.last_tree = None
        self.n_integrations = 0
        self.n_replans = 0

    def integrate(self, depth, cam_transform, ground_z):
        self.grid.integrate(
            depth=depth,
            cam_transform=cam_transform,
            ground_z=ground_z,
            hfov_deg=self.hfov_deg,
            max_range_m=self.max_range_m,
            min_range_m=self.min_range_m,
            obstacle_band_m=self.obstacle_band_m,
            pixel_stride=self.pixel_stride,
        )
        self.n_integrations += 1

    def abandon_goal(self):
        """The vehicle failed to finish its edge (stalled, or collided).

        Upstream has no equivalent -- a multirotor's trajectory tracker does
        not fail to reach a collision-checked point. Forget the remembered
        branch so the next tree is not re-seeded with the path the car just
        failed to drive. The history stays: it records where the car has been.
        """
        self.branch_memory = []
        self.current_goal = None

    def replan(self, vehicle_xy, vehicle_yaw):
        """Plan the next edge from the vehicle's pose (yaw in radians)."""
        t0 = time.perf_counter()
        vehicle_xy = (float(vehicle_xy[0]), float(vehicle_xy[1]))
        snapshot = self.grid.snapshot()
        checker = EdgeChecker(
            self.grid,
            snapshot,
            vehicle_xy,
            inflation_radius_m=self.inflation_radius_m,
            overshoot_m=self.overshoot_m,
            root_clearance_m=self.root_clearance_m,
        )
        tree = RrtTree(
            vehicle_xy,
            vehicle_yaw,
            self.grid,
            snapshot,
            checker,
            self.view,
            self.rng,
            extension_range_m=self.extension_range_m,
            degressive_coeff=self.degressive_coeff,
            zero_gain=self.zero_gain,
            bounds_margin_m=self.inflation_radius_m,
            forward_cone_deg=self.forward_cone_deg,
        )
        self.last_tree = tree
        tree.insert_branch(self.branch_memory)

        attempts = 0
        while not tree.gain_found() or tree.counter < self.initial_iterations:
            if tree.counter > self.cutoff_iterations:
                return self._backtrack(vehicle_xy, tree, attempts, "no_gain", t0)
            if attempts > 1000 * (tree.counter + 1):
                if tree.gain_found():
                    return self._commit(vehicle_xy, tree, attempts, "ok_capped", t0)
                return self._backtrack(vehicle_xy, tree, attempts, "sampling_exhausted", t0)
            tree.iterate()
            attempts += 1
        return self._commit(vehicle_xy, tree, attempts, "ok", t0)

    def _commit(self, vehicle_xy, tree, attempts, reason, t0):
        """Drive the first edge of the best branch (getBestEdge, memorizeBestBranch)."""
        child = tree.best_first_edge()
        goal = (child.x, child.y)
        self.history.append(vehicle_xy)
        self.branch_memory = tree.best_branch_states()
        self.current_goal = goal
        self.n_replans += 1
        return ReplanResult(
            waypoints=segment_waypoints(vehicle_xy, goal, self.waypoint_spacing_m),
            goal=goal,
            reason=reason,
            n_nodes=len(tree.nodes),
            n_attempts=attempts,
            best_gain=tree.best_gain,
            latency_s=time.perf_counter() - t0,
        )

    def _backtrack(self, vehicle_xy, tree, attempts, cause, t0):
        """getPathBackToPrevious: straight back to the root of the last edge.

        Like upstream, the return path is not collision-checked: it retraces
        an edge the vehicle has already driven.
        """
        stats = dict(
            n_nodes=len(tree.nodes),
            n_attempts=attempts,
            best_gain=tree.best_gain,
        )
        if not self.history:
            self.current_goal = None
            # No gain with nowhere left to go back to is upstream's shutdown.
            # A stalled sampler with no history is upstream's empty path,
            # which exploration.cpp answers by waiting 1s and asking again.
            reason = "finished" if cause == "no_gain" else "no_plan"
            return ReplanResult(reason=reason, latency_s=time.perf_counter() - t0, **stats)

        previous = self.history.pop()
        self.current_goal = previous
        return ReplanResult(
            waypoints=segment_waypoints(vehicle_xy, previous, self.waypoint_spacing_m),
            goal=previous,
            reason="backtrack_" + cause,
            backtrack=True,
            latency_s=time.perf_counter() - t0,
            **stats
        )
