"""
Offline checks for the Yamauchi baseline -- no CARLA server required.

Run with the CARLA env, from carla_client/:
    python -m baselines.yamauchi.test_offline

Covers the parts that are easy to get silently wrong: the intrinsics, the
depth backprojection and ray-casting, frontier detection, and Dijkstra
reachability/clearance. A synthetic scene with a known answer catches
coordinate-convention bugs that a CARLA run would only show as a subtly
wrong-looking map.
"""

import numpy as np

from .frontier import detect, frontier_mask
from .grid_map import FREE, OCC, UNKNOWN, GridMap, intrinsics_from_fov
from .explorer import YamauchiExplorer
from .planner import build_route, select_nearest, simplify, to_waypoints


class StubTransform(object):
    """Stands in for carla.Transform: only get_matrix() is used."""

    def __init__(self, matrix):
        self._m = np.asarray(matrix, dtype=np.float64)

    def get_matrix(self):
        return self._m


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("[%s] %s%s" % (status, label, ("  -- " + detail) if detail else ""))
    return bool(condition)


def test_intrinsics():
    K = intrinsics_from_fov(320, 240, 90.0)
    ok = check(
        "intrinsics: fx == fy == 160 for 320x240 @ 90deg hfov",
        abs(K[0, 0] - 160.0) < 1e-9 and abs(K[1, 1] - 160.0) < 1e-9,
        "got fx=%.3f fy=%.3f" % (K[0, 0], K[1, 1]),
    )
    vfov = 2.0 * np.degrees(np.arctan(120.0 / K[1, 1]))
    ok &= check(
        "intrinsics: derived vertical FOV is 73.7deg, not 90",
        abs(vfov - 73.74) < 0.1,
        "got %.2f deg" % vfov,
    )
    return ok


def synthetic_depth(width=320, height=240, hfov=90.0, wall_x=10.0, cam_height=1.7):
    """Depth of a scene: flat ground at z=-cam_height, wall at x=wall_x.

    Camera sits at the world origin looking down +x, level. CARLA's depth
    sensor reports *planar* depth (distance along the optical axis), which for
    this geometry equals the ray's x_cam.
    """
    K = intrinsics_from_fov(width, height, hfov)
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    vs, us = np.mgrid[0:height, 0:width].astype(np.float64)

    ny = (us - cx) / f  # +y right
    nz = -(vs - cy) / f  # +z up

    depth = np.full((height, width), wall_x, dtype=np.float32)
    looking_down = nz < -1e-9
    t_ground = np.where(looking_down, -cam_height / np.where(looking_down, nz, -1.0), np.inf)
    depth = np.minimum(depth, t_ground).astype(np.float32)
    return depth


def test_integration_geometry():
    grid = GridMap(min_x=-20.0, min_y=-25.0, max_x=40.0, max_y=25.0, resolution_m=0.5)
    depth = synthetic_depth()

    cam_at_origin_looking_x = np.eye(4)
    grid.integrate(
        depth=depth,
        cam_transform=StubTransform(cam_at_origin_looking_x),
        ground_z=-1.7,
        hfov_deg=90.0,
        max_range_m=40.0,
        obstacle_band_m=(0.5, 3.0),
    )
    snap = grid.snapshot()

    def at(x, y):
        ix, iy = grid.world_to_cell(x, y)
        return int(snap[int(ix), int(iy)])

    ok = check(
        "integration: road ahead of the camera is FREE",
        at(5.0, 0.0) == FREE,
        "cell state = %d" % at(5.0, 0.0),
    )
    ok &= check(
        "integration: wall at x=10 is OCC",
        at(10.0, 0.0) == OCC,
        "cell state = %d" % at(10.0, 0.0),
    )
    ok &= check(
        "integration: space behind the wall stays UNKNOWN",
        at(15.0, 0.0) == UNKNOWN,
        "cell state = %d" % at(15.0, 0.0),
    )
    ok &= check(
        "integration: space behind the camera stays UNKNOWN",
        at(-5.0, 0.0) == UNKNOWN,
        "cell state = %d" % at(-5.0, 0.0),
    )
    # A 90deg hfov cone reaches y=+-5 at x=5, so (5, 8) is outside the frustum.
    ok &= check(
        "integration: outside the 90deg frustum stays UNKNOWN",
        at(5.0, 8.0) == UNKNOWN,
        "cell state = %d" % at(5.0, 8.0),
    )

    # The same scene viewed with the env's wrong fy=120 would place the ground
    # return at a different range, bowing the plane. Confirm our K is the one
    # that puts the ground/wall break exactly at the wall.
    free_x = []
    for x in np.arange(1.0, 14.0, 0.5):
        if at(float(x), 0.0) == FREE:
            free_x.append(float(x))
    ok &= check(
        "integration: free space along the optical axis ends at the wall",
        bool(free_x) and 9.0 <= max(free_x) <= 10.0,
        "max free x = %s" % (max(free_x) if free_x else None),
    )
    return ok, grid, snap


def test_frontier_detection(grid, snap):
    mask = frontier_mask(snap)
    ok = check("frontier: mask is non-empty", bool(np.any(mask)))

    clusters = detect(grid, snapshot=snap, min_frontier_cells=4)
    ok &= check("frontier: at least one cluster found", len(clusters) > 0,
                "found %d" % len(clusters))

    if clusters:
        # Frontiers must sit on the edge of the observed wedge, never behind
        # the wall and never behind the camera.
        xs = [c.world[0] for c in clusters]
        ok &= check(
            "frontier: no cluster behind the wall (x > 11)",
            max(xs) <= 11.0,
            "max cluster x = %.2f" % max(xs),
        )
        for c in clusters:
            ix, iy = c.cell
            if snap[ix, iy] != FREE:
                ok &= check("frontier: representative cell is FREE", False,
                            "cluster %s sits on state %d" % (c, snap[ix, iy]))
                break
        else:
            ok &= check("frontier: every representative cell is FREE", True)
    return ok


def test_planner_reachability():
    """Corridor with a walled-off room: the room must be unreachable."""
    grid = GridMap(min_x=0.0, min_y=0.0, max_x=50.0, max_y=30.0, resolution_m=0.5)
    snap = np.full((grid.nx, grid.ny), UNKNOWN, dtype=np.int8)

    def fill(x0, x1, y0, y1, state):
        ix0, iy0 = grid.world_to_cell(x0, y0)
        ix1, iy1 = grid.world_to_cell(x1, y1)
        snap[int(ix0) : int(ix1), int(iy0) : int(iy1)] = state

    # A road-like corridor: free lane walled on both sides and behind, open
    # (bordering unknown) only at the far end. Walling the sides matters --
    # an unwalled corridor borders unknown along its whole length, so the
    # frontier becomes one ring enclosing the vehicle rather than a mouth
    # ahead of it.
    fill(1.5, 41.0, 9.5, 10.0, OCC)  # right wall
    fill(1.5, 41.0, 20.0, 20.5, OCC)  # left wall
    fill(1.5, 2.0, 9.5, 20.5, OCC)  # wall behind the vehicle
    fill(2.0, 40.0, 10.0, 20.0, FREE)
    # A sealed room beyond the corridor: free inside, fully ringed by obstacles.
    fill(41.0, 49.0, 8.0, 22.0, OCC)
    fill(42.0, 48.0, 10.0, 20.0, FREE)

    route = build_route(grid, snap, (5.0, 15.0), inflation_radius_m=1.0)
    ok = check("planner: route built from inside the corridor", route is not None)
    if route is None:
        return False

    far_corridor = grid.world_to_cell(38.0, 15.0)
    sealed_room = grid.world_to_cell(45.0, 15.0)
    ok &= check(
        "planner: far end of the corridor is reachable",
        np.isfinite(route.cost_to((int(far_corridor[0]), int(far_corridor[1])))),
    )
    ok &= check(
        "planner: sealed room is unreachable (infinite cost)",
        not np.isfinite(route.cost_to((int(sealed_room[0]), int(sealed_room[1])))),
    )

    # Path must stay clear of the inflated obstacles.
    cells = route.path_to((int(far_corridor[0]), int(far_corridor[1])))
    ok &= check("planner: non-empty path to the corridor end", len(cells) > 2,
                "%d cells" % len(cells))
    if cells:
        clear = all(route.passable[c[0], c[1]] for c in cells)
        ok &= check("planner: every path cell clears the inflated obstacles", clear)

        simple = simplify(cells, route.passable)
        ok &= check(
            "planner: shortcutting shortens the straight corridor path",
            len(simple) < len(cells),
            "%d -> %d cells" % (len(cells), len(simple)),
        )
        ok &= check(
            "planner: simplified path also clears obstacles",
            all(route.passable[c[0], c[1]] for c in simple),
        )

        # A ~33m corridor at 2m spacing must yield ~17 waypoints, not one
        # target at the far end (which would starve the PID of steering).
        wps = to_waypoints(grid, simple, spacing_m=2.0)
        ok &= check(
            "planner: straight corridor is resampled, not collapsed",
            len(wps) >= 14,
            "%d waypoints over ~33m" % len(wps),
        )
        if len(wps) > 1:
            gaps = [
                np.hypot(wps[i][0] - wps[i - 1][0], wps[i][1] - wps[i - 1][1])
                for i in range(1, len(wps))
            ]
            ok &= check(
                "planner: no waypoint gap exceeds the requested spacing",
                max(gaps) <= 2.0 + 1e-6,
                "max gap = %.3f m" % max(gaps),
            )

    # Selection must travel to the far unexplored end, not to a frontier the
    # vehicle is already sitting on.
    clusters = detect(grid, snapshot=snap, min_frontier_cells=2)
    if clusters:
        goal, gcells = select_nearest(route, clusters, grid)
        ok &= check(
            "planner: selected goal is reachable",
            goal is not None and len(gcells) > 0,
            "goal = %s" % (goal,),
        )
        if gcells:
            gx, gy = grid.cell_to_world(gcells[-1][0], gcells[-1][1])
            ok &= check(
                "planner: goal is the far corridor mouth, not the vehicle's cell",
                float(gx) > 30.0,
                "goal routed to x = %.2f" % float(gx),
            )
    return ok


def test_min_goal_distance():
    """The forward-camera blind spot must not trap the vehicle on itself.

    An unwalled free patch borders unknown space on every side, so without a
    minimum goal distance the nearest frontier cell is the one the vehicle is
    standing on and it never travels. This is the real failure mode of a
    90-degree forward-facing camera, not a synthetic curiosity.
    """
    grid = GridMap(min_x=0.0, min_y=0.0, max_x=60.0, max_y=40.0, resolution_m=0.5)
    snap = np.full((grid.nx, grid.ny), UNKNOWN, dtype=np.int8)
    ix0, iy0 = grid.world_to_cell(5.0, 5.0)
    ix1, iy1 = grid.world_to_cell(50.0, 35.0)
    snap[int(ix0) : int(ix1), int(iy0) : int(iy1)] = FREE

    route = build_route(grid, snap, (10.0, 20.0), inflation_radius_m=1.0)
    clusters = detect(grid, snapshot=snap, min_frontier_cells=2)

    def path_length_m(cells):
        """Length along the cell path -- the quantity the guard constrains.

        Not the straight-line distance to the goal: a path that bends around
        an inflated obstacle is legitimately longer than the crow-flies gap.
        """
        if len(cells) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(cells)):
            total += np.hypot(
                cells[i][0] - cells[i - 1][0], cells[i][1] - cells[i - 1][1]
            )
        return total * grid.res

    _, cells_near = select_nearest(route, clusters, grid, min_goal_distance_m=0.0)
    ok = check(
        "min_goal_distance: without the guard the goal is on top of the vehicle",
        path_length_m(cells_near) < 8.0,
        "path to goal = %.2f m" % path_length_m(cells_near),
    )

    goal_far, cells_far = select_nearest(route, clusters, grid, min_goal_distance_m=8.0)
    ok &= check(
        "min_goal_distance: with the guard the goal is at least 8m of path away",
        goal_far is not None and path_length_m(cells_far) >= 8.0,
        "path to goal = %.2f m" % path_length_m(cells_far),
    )
    return ok


class StubSpawnPoint(object):
    """Stands in for a carla.Transform spawn point: only .location.x/.y are used."""

    def __init__(self, x, y):
        self.location = type("Loc", (object,), {"x": x, "y": y})()


class StubCarlaMap(object):
    """Stands in for carla.Map: only get_spawn_points() is used, to size the grid."""

    def __init__(self, min_xy, max_xy):
        self._pts = [StubSpawnPoint(*min_xy), StubSpawnPoint(*max_xy)]

    def get_spawn_points(self):
        return self._pts


def test_forward_cone():
    """Lane open at both ends, vehicle nearer the -x end: the nearest frontier is behind."""
    grid = GridMap(min_x=0.0, min_y=0.0, max_x=60.0, max_y=30.0, resolution_m=0.5)
    snap = np.full((grid.nx, grid.ny), UNKNOWN, dtype=np.int8)

    def fill(arr, x0, x1, y0, y1, state):
        ix0, iy0 = grid.world_to_cell(x0, y0)
        ix1, iy1 = grid.world_to_cell(x1, y1)
        arr[int(ix0) : int(ix1), int(iy0) : int(iy1)] = state

    fill(snap, 8.0, 52.0, 9.5, 10.0, OCC)
    fill(snap, 8.0, 52.0, 20.0, 20.5, OCC)
    fill(snap, 10.0, 50.0, 10.0, 20.0, FREE)
    vehicle = (22.0, 15.0)
    route = build_route(grid, snap, vehicle, inflation_radius_m=1.0)
    clusters = detect(grid, snapshot=snap, min_frontier_cells=2)

    def goal_x(**kwargs):
        _, cells = select_nearest(route, clusters, grid, vehicle_xy=vehicle, **kwargs)
        return float(grid.cell_to_world(cells[-1][0], cells[-1][1])[0]) if cells else None

    x = goal_x()
    ok = check("cone: without it, the nearest frontier is the one behind (x=10 end)",
               x is not None and x < 22.0, "goal x = %s" % x)
    x = goal_x(vehicle_yaw_deg=0.0, forward_cone_deg=90.0)
    ok &= check("cone: facing +x, the goal is the frontier ahead (x=50 end)",
                x is not None and x > 40.0, "goal x = %s" % x)
    x = goal_x(vehicle_yaw_deg=180.0, forward_cone_deg=90.0)
    ok &= check("cone: facing -x, the goal is the x=10 end",
                x is not None and x < 22.0, "goal x = %s" % x)

    # Seal the far end: nothing is left ahead of a vehicle facing +x.
    sealed = snap.copy()
    fill(sealed, 50.0, 50.5, 9.5, 20.5, OCC)
    ex = YamauchiExplorer(
        StubCarlaMap((0.0, 0.0), (60.0, 30.0)),
        resolution_m=0.5,
        margin_m=0.0,
        min_frontier_cells=2,
        inflation_radius_m=1.0,
        min_obstacle_cells=1,
        forward_cone_deg=90.0,
    )
    ex.grid.log_odds[...] = np.where(sealed == FREE, -1.0, np.where(sealed == OCC, 1.0, 0.0))
    res = ex.replan(vehicle, vehicle_yaw_deg=0.0)
    ok &= check("cone: with nothing reachable ahead, falls back to the frontier behind",
                res.reason == "ok" and res.behind and res.goal[0] < 22.0,
                "%r, behind=%s, goal=%s" % (res, res.behind, res.goal))
    res = ex.replan(vehicle, vehicle_yaw_deg=180.0)
    ok &= check("cone: a goal inside the cone is not flagged as behind",
                res.reason == "ok" and not res.behind, "%r, behind=%s" % (res, res.behind))
    return ok


def main():
    print("=" * 70)
    print("Yamauchi baseline -- offline checks")
    print("=" * 70)
    results = []

    results.append(test_intrinsics())
    ok, grid, snap = test_integration_geometry()
    results.append(ok)
    results.append(test_frontier_detection(grid, snap))
    results.append(test_planner_reachability())
    results.append(test_min_goal_distance())
    results.append(test_forward_cone())

    print("=" * 70)
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
