"""
Offline checks for the NBVP baseline -- no CARLA server required.

Run with the CARLA env, from carla_client/:
    python -m baselines.nbvp.test_offline

Mapping is the Yamauchi package's and is covered by yamauchi.test_offline.
These checks cover what this package adds: the 2D view gain, the edge
collision check, the tree, the planning loop's stopping rules, and replan
latency on a town-sized grid.
"""

import time

import numpy as np

from baselines.yamauchi.grid_map import FREE, OCC, UNKNOWN, GridMap

from .collision import EdgeChecker
from .explorer import NBVPExplorer, segment_waypoints
from .views import ViewModel

RES = 0.25


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("[%s] %s%s" % (status, label, ("  -- " + detail) if detail else ""))
    return bool(condition)


def blank(width, height):
    grid = GridMap(0.0, 0.0, width, height, resolution_m=RES, min_obstacle_cells=1)
    return grid, np.full((grid.nx, grid.ny), UNKNOWN, dtype=np.int8)


def fill(grid, snap, x0, x1, y0, y1, state):
    ix0, iy0 = grid.world_to_cell(x0, y0)
    ix1, iy1 = grid.world_to_cell(x1, y1)
    snap[int(ix0) : int(ix1), int(iy0) : int(iy1)] = state


def load(grid, snap):
    """Write a tri-state map into the grid's log-odds so snapshot() returns it."""
    grid.log_odds[...] = np.where(snap == FREE, -1.0, np.where(snap == OCC, 1.0, 0.0))


def corridor(open_end_x=40.0, length=78.0):
    """Walled lane along +x: known FREE up to open_end_x, UNKNOWN beyond."""
    grid, snap = blank(80.0, 30.0)
    fill(grid, snap, 1.5, length, 9.5, 10.0, OCC)
    fill(grid, snap, 1.5, length, 20.0, 20.5, OCC)
    fill(grid, snap, 1.5, 2.0, 9.5, 20.5, OCC)
    fill(grid, snap, 2.0, open_end_x, 10.0, 20.0, FREE)
    return grid, snap


# ── view gain ─────────────────────────────────────────────────────────────
def test_view_gain():
    view = ViewModel(RES, hfov_deg=90.0, vfov_deg=73.74, gain_range_m=12.0,
                     cam_height_m=1.7, cam_forward_m=1.0)
    ok = check(
        "gain: ground enters a 1.7m camera's 73.7deg vertical FOV at 2.27m",
        abs(view.min_ground_range_m - 2.267) < 0.01,
        "got %.3f m" % view.min_ground_range_m,
    )

    grid, snap = blank(60.0, 60.0)
    wedge = np.pi / 4.0 * (12.0 ** 2 - view.min_ground_range_m ** 2)
    g_x = view.gain(grid, snap, 20.0, 30.0, 0.0)
    ok &= check(
        "gain: open unknown field scores the 90deg wedge area",
        abs(g_x - wedge) / wedge < 0.08,
        "got %.1f m2, wedge %.1f m2" % (g_x, wedge),
    )
    g_y = view.gain(grid, snap, 30.0, 20.0, np.pi / 2.0)
    ok &= check(
        "gain: heading +y scores the same as heading +x",
        abs(g_y - g_x) / g_x < 0.03,
        "%.1f vs %.1f m2" % (g_y, g_x),
    )

    cells = view.visible_unknown(grid, snap, 20.0, 30.0, 0.0)
    cx, _ = grid.cell_to_world(cells // grid.ny, cells % grid.ny)
    ok &= check(
        "gain: nothing counted behind the camera",
        float(cx.min()) > 21.0 + view.min_ground_range_m * np.cos(np.pi / 4.0) - RES,
        "min x = %.2f" % float(cx.min()),
    )

    free = np.full_like(snap, FREE)
    ok &= check("gain: fully mapped view scores zero",
                view.gain(grid, free, 20.0, 30.0, 0.0) == 0.0)

    near = np.full_like(snap, FREE)
    fill(grid, near, 20.0, 23.0, 29.0, 31.0, UNKNOWN)
    ok &= check(
        "gain: unknown below the image (inside min ground range) scores zero",
        view.gain(grid, near, 20.0, 30.0, 0.0) == 0.0,
    )

    # Wall 6m ahead of the camera across the whole view.
    walled = snap.copy()
    fill(grid, walled, 27.0, 27.5, 0.0, 60.0, OCC)
    cells = view.visible_unknown(grid, walled, 20.0, 30.0, 0.0)
    cx, _ = grid.cell_to_world(cells // grid.ny, cells % grid.ny)
    ok &= check("gain: nothing counted behind a wall", float(cx.max()) < 27.0,
                "max x = %.2f" % float(cx.max()))
    # Triangle |y| <= x for x in [0, 6] from the camera, minus the sector
    # below the image.
    bound = 36.0 - np.pi / 4.0 * view.min_ground_range_m ** 2
    g_wall = view.gain(grid, walled, 20.0, 30.0, 0.0)
    ok &= check(
        "gain: wall truncates the wedge to the area in front of it",
        0.8 * bound < g_wall < 1.15 * bound,
        "got %.1f m2, expected ~%.1f" % (g_wall, bound),
    )

    t0 = time.perf_counter()
    for _ in range(100):
        view.gain(grid, snap, 20.0, 30.0, 0.3)
    per = (time.perf_counter() - t0) / 100.0
    ok &= check("gain: under 5 ms per node", per < 0.005, "%.2f ms" % (per * 1e3))
    return ok


# ── collision check ───────────────────────────────────────────────────────
def test_edge_checker():
    grid, snap = corridor(open_end_x=50.0)
    root = (6.0, 15.0)
    chk = EdgeChecker(grid, snap, root, inflation_radius_m=1.5, overshoot_m=1.0, root_clearance_m=4.0)

    ok = check("collision: straight edge down the lane is clear",
               chk.edge_clear(10.0, 15.0, 18.0, 15.0))
    ok &= check("collision: edge through the lane wall is blocked",
                not chk.edge_clear(10.0, 15.0, 10.0, 21.0))
    ok &= check("collision: edge inside the wall's inflation band is blocked",
                not chk.edge_clear(10.0, 11.0, 18.0, 11.0))
    ok &= check("collision: edge into UNKNOWN is blocked (tree grows only through known FREE)",
                not chk.edge_clear(44.0, 15.0, 52.0, 15.0))

    walled = snap.copy()
    fill(grid, walled, 30.0, 30.5, 10.0, 20.0, OCC)
    no_over = EdgeChecker(grid, walled, root, 1.5, overshoot_m=0.0, root_clearance_m=4.0)
    over = EdgeChecker(grid, walled, root, 1.5, overshoot_m=1.0, root_clearance_m=4.0)
    ok &= check("collision: edge ending 2.2m short of a wall is clear without overshoot",
                no_over.edge_clear(20.0, 15.0, 27.8, 15.0))
    ok &= check("collision: ...and blocked once the 1m overshoot is checked",
                not over.edge_clear(20.0, 15.0, 27.8, 15.0))

    # Forward-camera blind zone: the vehicle's surroundings were never observed.
    blind = snap.copy()
    fill(grid, blind, 2.0, 10.0, 10.0, 20.0, UNKNOWN)
    relaxed = EdgeChecker(grid, blind, root, 1.5, 1.0, root_clearance_m=4.0)
    strict = EdgeChecker(grid, blind, root, 1.5, 1.0, root_clearance_m=0.0)
    ok &= check("collision: root clearance lets an edge leave the unobserved blind zone",
                relaxed.edge_clear(6.0, 15.0, 13.0, 15.0))
    ok &= check("collision: without it the same edge is blocked",
                not strict.edge_clear(6.0, 15.0, 13.0, 15.0))
    ok &= check("collision: root clearance never lets a node sit in unobserved space",
                not relaxed.edge_clear(6.0, 15.0, 8.0, 15.0))
    fill(grid, blind, 8.0, 8.5, 14.5, 15.5, OCC)
    relaxed = EdgeChecker(grid, blind, root, 1.5, 1.0, root_clearance_m=4.0)
    ok &= check("collision: root clearance never waives a real obstacle",
                not relaxed.edge_clear(6.0, 15.0, 13.0, 15.0))
    return ok


# ── tree and planning loop ────────────────────────────────────────────────
def make_explorer(grid, snap, seed=1, **kwargs):
    load(grid, snap)
    return NBVPExplorer(grid, seed=seed, **kwargs)


def test_replan_corridor():
    """Lane mapped to x=40, unknown beyond: NBVP must head for the open end."""
    grid, snap = corridor(open_end_x=40.0)
    ex = make_explorer(grid, snap)
    res = ex.replan((6.0, 15.0), 0.0)
    tree = ex.last_tree

    ok = check("replan: succeeds in a mapped lane", res.reason == "ok", repr(res))
    if res.reason != "ok":
        return False
    ok &= check("replan: at least root + initial_iterations nodes",
                len(tree.nodes) >= 16, "%d nodes" % len(tree.nodes))
    ok &= check("replan: first edge heads down the lane (+x)", res.goal[0] > 6.0,
                "goal = (%.1f, %.1f)" % res.goal)
    best = tree.best_node
    cells = ex.view.visible_unknown(grid, grid.snapshot(), best.x, best.y, best.yaw)
    cx, _ = grid.cell_to_world(cells // grid.ny, cells % grid.ny)
    ok &= check("replan: best viewpoint's gain is the unknown end of the lane (x > 40)",
                cells.size > 0 and float(cx.min()) > 40.0,
                "best node (%.1f, %.1f) yaw=%.0fdeg sees %d cells, min x %.1f"
                % (best.x, best.y, np.degrees(best.yaw), cells.size,
                   float(cx.min()) if cells.size else float("nan")))

    inside = all(2.0 < n.x < 40.0 and 10.0 < n.y < 20.0 for n in tree.nodes)
    ok &= check("replan: every node lies in the mapped lane", inside)
    longest = max(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in tree.edges())
    ok &= check("replan: no edge exceeds extension_range", longest <= 8.0 + 1e-6,
                "longest %.2f m" % longest)

    wps = res.waypoints
    gaps = [np.hypot(wps[0][0] - 6.0, wps[0][1] - 15.0)] + [
        np.hypot(wps[i][0] - wps[i - 1][0], wps[i][1] - wps[i - 1][1]) for i in range(1, len(wps))
    ]
    ok &= check("replan: waypoints end at the edge goal, no gap over 2m",
                np.allclose(wps[-1], res.goal) and max(gaps) <= 2.0 + 1e-6,
                "%d waypoints, max gap %.2f" % (len(wps), max(gaps)))
    ok &= check("replan: root pushed onto the history", ex.history == [(6.0, 15.0)])

    # Receding horizon: drive the edge, replan, and the remembered branch must
    # be re-inserted into the new tree.
    memory = list(ex.branch_memory)
    ex.replan(res.goal, float(np.arctan2(res.goal[1] - 15.0, res.goal[0] - 6.0)))
    if memory:
        new_xy = np.array([(n.x, n.y) for n in ex.last_tree.nodes])
        found = all(np.min(np.hypot(new_xy[:, 0] - x, new_xy[:, 1] - y)) < 1e-6 for x, y in memory)
        ok &= check("replan: previous best branch re-inserted into the next tree", found,
                    "%d remembered nodes" % len(memory))
    else:
        ok &= check("replan: best branch was a single edge (nothing to remember)", True)

    a = make_explorer(*corridor(open_end_x=40.0), seed=7).replan((6.0, 15.0), 0.0)
    b = make_explorer(*corridor(open_end_x=40.0), seed=7).replan((6.0, 15.0), 0.0)
    ok &= check("replan: same seed, same map -> same edge", a.goal == b.goal)
    return ok


def test_blind_start():
    """Spawn state: the vehicle's own surroundings are UNKNOWN."""
    grid, snap = corridor(open_end_x=40.0)
    fill(grid, snap, 2.0, 9.0, 10.0, 20.0, UNKNOWN)
    res = make_explorer(grid, snap).replan((6.0, 15.0), 0.0)
    return check("replan: plans out of the unobserved blind zone at spawn",
                 res.reason == "ok", repr(res))


def test_forward_cone():
    """Open mapped field ringed by unknown: gain is available in every direction."""
    grid, snap = blank(80.0, 80.0)
    fill(grid, snap, 10.0, 70.0, 10.0, 70.0, FREE)
    root = (40.0, 40.0)

    def off_heading_deg(node, yaw):
        dx, dy = node.x - root[0], node.y - root[1]
        return abs(np.degrees((np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi))

    ex = make_explorer(grid, snap)
    ex.replan(root, 0.0)
    worst = max(off_heading_deg(n, 0.0) for n in ex.last_tree.nodes[1:])
    ok = check("cone off (upstream): tree grows behind the vehicle too", worst > 90.0,
               "widest node %.0fdeg off heading" % worst)

    for cone, yaw in ((90.0, 0.0), (45.0, np.pi / 2.0)):
        ex = make_explorer(grid, snap, forward_cone_deg=cone)
        res = ex.replan(root, yaw)
        worst = max(off_heading_deg(n, yaw) for n in ex.last_tree.nodes[1:])
        ok &= check("cone %.0fdeg: every viewpoint within the cone, plan still found" % cone,
                    res.reason == "ok" and worst <= cone + 1e-6,
                    "%s, widest node %.1fdeg off heading" % (res.reason, worst))
    return ok


def test_no_gain():
    """Sealed, fully mapped room: no viewpoint sees anything unknown."""
    grid, snap = blank(40.0, 30.0)
    fill(grid, snap, 10.0, 30.0, 10.0, 20.0, OCC)
    fill(grid, snap, 10.5, 29.5, 10.5, 19.5, FREE)

    ex = make_explorer(grid, snap)
    res = ex.replan((20.0, 15.0), 0.0)
    ok = check("no gain, empty history: reports finished (upstream's shutdown)",
               res.reason == "finished", repr(res))
    ok &= check("no gain: gave up only past cutoff_iterations nodes", res.n_nodes > 200,
                "%d nodes" % res.n_nodes)

    ex = make_explorer(grid, snap)
    ex.history = [(15.0, 15.0), (18.0, 15.0)]
    res = ex.replan((20.0, 15.0), 0.0)
    ok &= check("no gain, with history: backtracks instead of shutting down",
                res.reason == "backtrack_no_gain" and res.backtrack, repr(res))
    ok &= check("backtrack: heads to the most recent previous root",
                bool(res.waypoints) and np.allclose(res.waypoints[-1], (18.0, 15.0)))
    ok &= check("backtrack: pops exactly one history entry", ex.history == [(15.0, 15.0)])
    return ok


def test_capped_with_gain():
    """Spawn state: a town-sized map with only 12 m of lane mapped ahead.

    On a town-sized map nearly every sample is far away, so every edge is the
    full 8 m and only one or two fit in the mapped stub -- never 15. They do
    see the unmapped lane beyond it.
    """
    grid, snap = blank(400.0, 400.0)
    fill(grid, snap, 200.0, 230.0, 194.5, 195.0, OCC)
    fill(grid, snap, 200.0, 230.0, 205.0, 205.5, OCC)
    fill(grid, snap, 200.0, 200.5, 194.5, 205.5, OCC)
    fill(grid, snap, 200.5, 214.0, 195.0, 205.0, FREE)
    ex = make_explorer(grid, snap)
    res = ex.replan((204.0, 200.0), 0.0)
    ok = check("stalled tree with gain: drives its best edge instead of giving up",
               res.reason == "ok_capped" and bool(res.waypoints) and res.goal[0] > 204.0
               and res.n_nodes < 16 and res.best_gain > 0.0, repr(res))
    ok &= check("stalled tree with gain: a real plan (root pushed onto the history)",
                ex.history == [(204.0, 200.0)])
    return ok


def test_sampling_exhausted():
    """Nothing known beyond the root clearance: no edge can ever be added."""
    grid, snap = blank(60.0, 60.0)
    ex = make_explorer(grid, snap)
    res = ex.replan((30.0, 30.0), 0.0)
    ok = check("stalled sampler, empty history: empty plan to retry (not finished)",
               res.reason == "no_plan", repr(res))
    ex.history = [(25.0, 30.0)]
    res = ex.replan((30.0, 30.0), 0.0)
    ok &= check("stalled sampler, with history: backtracks (upstream behaviour)",
                res.reason == "backtrack_sampling_exhausted", repr(res))
    return ok


def test_segment_waypoints():
    wps = segment_waypoints((0.0, 0.0), (7.0, 0.0), spacing_m=2.0)
    ok = check("waypoints: 7m edge at 2m spacing -> 4 evenly spaced points",
               len(wps) == 4 and np.allclose(wps[-1], (7.0, 0.0)), str(wps))
    ok &= check("waypoints: zero-length edge -> none",
                segment_waypoints((1.0, 1.0), (1.0, 1.0)) == [])
    return ok


# ── latency on a town-sized grid ──────────────────────────────────────────
def road_network(known_radius_m=None, center=(150.0, 150.0)):
    """600 x 550 m grid of 10m roads every 100m, walled by building fronts."""
    grid, snap = blank(600.0, 550.0)
    xs = np.arange(50.0, 600.0, 100.0)
    ys = np.arange(50.0, 550.0, 100.0)
    for c in xs:
        fill(grid, snap, c - 5.5, c - 5.0, 0.0, 550.0, OCC)
        fill(grid, snap, c + 5.0, c + 5.5, 0.0, 550.0, OCC)
    for c in ys:
        fill(grid, snap, 0.0, 600.0, c - 5.5, c - 5.0, OCC)
        fill(grid, snap, 0.0, 600.0, c + 5.0, c + 5.5, OCC)
    for c in xs:
        fill(grid, snap, c - 5.0, c + 5.0, 0.0, 550.0, FREE)
    for c in ys:
        fill(grid, snap, 0.0, 600.0, c - 5.0, c + 5.0, FREE)
    if known_radius_m is not None:
        wx, wy = grid.cell_to_world(np.arange(grid.nx)[:, None], np.arange(grid.ny)[None, :])
        far = np.hypot(wx - center[0], wy - center[1]) > known_radius_m
        snap[far] = UNKNOWN
    return grid, snap


def test_latency():
    grid, snap = road_network(known_radius_m=80.0)
    ex = make_explorer(grid, snap)
    t0 = time.perf_counter()
    res = ex.replan((150.0, 150.0), 0.0)
    typical = time.perf_counter() - t0
    ok = check("latency: typical replan on a %dx%d grid under 3s" % (grid.nx, grid.ny),
               res.reason == "ok" and typical < 3.0,
               "%.2fs, %d nodes, %s" % (typical, res.n_nodes, res.reason))

    grid, snap = road_network()
    ex = make_explorer(grid, snap)
    t0 = time.perf_counter()
    res = ex.replan((150.0, 150.0), 0.0)
    worst = time.perf_counter() - t0
    ok &= check("latency: worst case (fully mapped, runs to cutoff) under 10s",
                res.reason == "finished" and worst < 10.0,
                "%.2fs, %d nodes, %d attempts, %s" % (worst, res.n_nodes, res.n_attempts, res.reason))
    return ok


def main():
    print("=" * 70)
    print("NBVP baseline -- offline checks")
    print("=" * 70)
    results = [
        test_view_gain(),
        test_edge_checker(),
        test_replan_corridor(),
        test_blind_start(),
        test_forward_cone(),
        test_no_gain(),
        test_capped_with_gain(),
        test_sampling_exhausted(),
        test_segment_waypoints(),
        test_latency(),
    ]
    print("=" * 70)
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
