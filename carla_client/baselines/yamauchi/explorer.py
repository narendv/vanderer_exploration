"""
YamauchiExplorer -- the whole classical baseline behind one object.

Holds the persistent map and blacklist across the run (including across
collision respawns: exploration knowledge is cumulative and the coverage
metric spans the whole run, so only the Dijkstra source moves on reset).

The controller calls integrate() every simulator step and replan() every N
steps; nothing else in this package needs to be touched from outside.
"""

import time

import numpy as np

from .frontier import Blacklist, detect
from .grid_map import GridMap
from .planner import build_route, select_nearest, simplify, to_waypoints


class ReplanResult(object):
    __slots__ = (
        "waypoints", "goal", "n_clusters", "n_reachable", "latency_s", "reason", "behind"
    )

    def __init__(
        self,
        waypoints=None,
        goal=None,
        n_clusters=0,
        n_reachable=0,
        latency_s=0.0,
        reason="ok",
        behind=False,
    ):
        self.waypoints = waypoints or []
        self.goal = goal
        self.n_clusters = n_clusters
        self.n_reachable = n_reachable
        self.latency_s = latency_s
        self.reason = reason
        # True when no frontier was reachable inside the forward cone and the
        # goal had to be taken from behind the vehicle.
        self.behind = behind

    def __repr__(self):
        return "ReplanResult(%s, n_wp=%d, clusters=%d)" % (
            self.reason,
            len(self.waypoints),
            self.n_clusters,
        )


class YamauchiExplorer(object):
    def __init__(
        self,
        carla_map,
        resolution_m=0.5,
        margin_m=100.0,
        hfov_deg=90.0,
        max_range_m=40.0,
        min_range_m=1.0,
        obstacle_band_m=(0.5, 3.0),
        pixel_stride=2,
        min_frontier_cells=4,
        inflation_radius_m=1.5,
        waypoint_spacing_m=2.0,
        blacklist_radius_m=6.0,
        min_goal_distance_m=8.0,
        min_obstacle_cells=4,
        forward_cone_deg=0.0,
    ):
        self.grid = GridMap.from_carla_map(
            carla_map,
            resolution_m=resolution_m,
            margin_m=margin_m,
            min_obstacle_cells=min_obstacle_cells,
        )
        self.blacklist = Blacklist(radius_m=blacklist_radius_m)

        self.hfov_deg = hfov_deg
        self.max_range_m = max_range_m
        self.min_range_m = min_range_m
        self.obstacle_band_m = obstacle_band_m
        self.pixel_stride = pixel_stride
        self.min_frontier_cells = min_frontier_cells
        self.inflation_radius_m = inflation_radius_m
        self.waypoint_spacing_m = waypoint_spacing_m
        self.min_goal_distance_m = min_goal_distance_m
        self.forward_cone_deg = forward_cone_deg

        self.current_goal = None
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
        """Blacklist the current goal so the greedy rule stops re-picking it."""
        if self.current_goal is not None:
            self.blacklist.add(self.current_goal)
            self.current_goal = None

    def replan(self, vehicle_xy, vehicle_yaw_deg=None):
        """Plan to the nearest frontier. The forward cone needs vehicle_yaw_deg."""
        t0 = time.perf_counter()
        snapshot = self.grid.snapshot()

        clusters = detect(
            self.grid, snapshot=snapshot, min_frontier_cells=self.min_frontier_cells
        )
        n_all = len(clusters)
        clusters = self.blacklist.filter(clusters)
        if not clusters:
            return ReplanResult(
                n_clusters=n_all,
                latency_s=time.perf_counter() - t0,
                reason="no_frontiers" if n_all == 0 else "all_blacklisted",
            )

        route = build_route(
            self.grid,
            snapshot,
            vehicle_xy,
            inflation_radius_m=self.inflation_radius_m,
        )
        if route is None:
            return ReplanResult(
                n_clusters=n_all,
                latency_s=time.perf_counter() - t0,
                reason="no_route_source",
            )

        goal, cells = select_nearest(
            route,
            clusters,
            self.grid,
            min_goal_distance_m=self.min_goal_distance_m,
            vehicle_xy=vehicle_xy,
            vehicle_yaw_deg=vehicle_yaw_deg,
            forward_cone_deg=self.forward_cone_deg,
        )
        behind = False
        if (goal is None or len(cells) < 2) and self.forward_cone_deg > 0:
            # Nothing reachable ahead. Turn around rather than stall: the cone
            # is a preference for frontiers ahead, not a ban on the rest --
            # NBVP's cone falls back to backtracking the same way.
            goal, cells = select_nearest(
                route,
                clusters,
                self.grid,
                min_goal_distance_m=self.min_goal_distance_m,
            )
            behind = goal is not None and len(cells) >= 2
        if goal is None or len(cells) < 2:
            return ReplanResult(
                n_clusters=n_all,
                latency_s=time.perf_counter() - t0,
                reason="no_reachable_frontier",
            )

        cells = simplify(cells, route.passable)
        waypoints = to_waypoints(
            self.grid, cells, spacing_m=self.waypoint_spacing_m
        )
        if not waypoints:
            return ReplanResult(
                n_clusters=n_all,
                latency_s=time.perf_counter() - t0,
                reason="empty_path",
            )

        n_reachable = int(np.count_nonzero(np.isfinite(route.dist)))
        self.current_goal = goal.world
        self.n_replans += 1
        return ReplanResult(
            waypoints=waypoints,
            goal=goal.world,
            n_clusters=n_all,
            n_reachable=n_reachable,
            latency_s=time.perf_counter() - t0,
            reason="ok",
            behind=behind,
        )
