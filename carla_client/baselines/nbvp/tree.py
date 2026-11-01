"""
Rapidly-exploring random tree of viewpoints -- nbvplanner's RrtTree
(nbvplanner/src/rrt.cpp) on the BEV grid.

Kept from upstream:
  * samples are drawn uniformly from a disc centred on the root whose radius
    is the diagonal of the exploration bounds, and rejected if they fall
    outside the bounds shrunk by the vehicle half-size (iterate(),
    softBounds=false). Sampling the whole disc rather than the box keeps the
    tree from being biased towards the centre of the map
  * a sample extends the nearest node by at most extension_range_m, and the
    edge must pass the collision check
  * a node's gain is its parent's gain plus its own view gain discounted by
    exp(-degressive_coeff * path distance from the root)
  * `counter` counts nodes added, not sampling attempts -- the planning loop's
    stopping rules are written in terms of it
  * the remainder of the previous best branch is re-inserted before any
    sampling (initialize()), re-checked for collision and re-scored on the
    current map

Changed for a car:
  * state is (x, y, yaw); z is dropped
  * yaw is the heading of the edge into the node, not sampled. Upstream draws
    a random yaw per node because a multirotor can point its camera anywhere
    at any position. The car's camera faces forward, so it looks where it
    has just driven.
  * optionally (forward_cone_deg > 0), a node must lie within that angle of
    the vehicle's heading, seen from the root. The forward camera leaves the
    space behind the car unknown, so unconstrained NBVP keeps choosing
    viewpoints behind it, and the car cannot make the U-turns that requires.
    0 disables the cone, as upstream.
"""

import math

import numpy as np


class Node(object):
    __slots__ = ("x", "y", "yaw", "parent", "distance", "view_gain", "gain")

    def __init__(self, x, y, yaw, parent=None, distance=0.0, view_gain=0.0, gain=0.0):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.parent = parent
        self.distance = float(distance)
        self.view_gain = float(view_gain)
        self.gain = float(gain)


class RrtTree(object):
    def __init__(
        self,
        root_xy,
        root_yaw,
        grid,
        snapshot,
        checker,
        view,
        rng,
        extension_range_m=8.0,
        degressive_coeff=0.0625,
        zero_gain=0.0,
        bounds_margin_m=1.5,
        forward_cone_deg=0.0,
    ):
        self.grid = grid
        self.snapshot = snapshot
        self.checker = checker
        self.view = view
        self.rng = rng
        self.extension_range_m = float(extension_range_m)
        self.degressive_coeff = float(degressive_coeff)
        self.zero_gain = float(zero_gain)

        self.root = Node(root_xy[0], root_xy[1], root_yaw, gain=zero_gain)
        self.cone_cos = math.cos(math.radians(forward_cone_deg)) if forward_cone_deg > 0 else None
        self.nodes = [self.root]
        self._xy = np.zeros((64, 2), dtype=np.float64)
        self._xy[0] = (self.root.x, self.root.y)

        self.counter = 0
        self.best_gain = self.zero_gain
        self.best_node = None

        m = float(bounds_margin_m)
        self.lo = np.array([grid.origin_x + m, grid.origin_y + m])
        self.hi = np.array(
            [grid.origin_x + grid.nx * grid.res - m, grid.origin_y + grid.ny * grid.res - m]
        )
        self.sample_radius = float(np.hypot(*(self.hi - self.lo)))

    # ── growth ────────────────────────────────────────────────────────────
    def gain_found(self):
        return self.best_gain > self.zero_gain

    def iterate(self):
        """One sampling attempt (upstream iterate(1)). The new node, or None."""
        x, y = self._sample()
        return self._extend(x, y)

    def insert_branch(self, states):
        """Re-insert the previous best branch, shallowest node first."""
        for x, y in states:
            self._extend(x, y)

    def _sample(self):
        r = self.sample_radius
        root = np.array([self.root.x, self.root.y])
        while True:
            pts = self.rng.uniform(-r, r, size=(64, 2))
            pts = pts[(pts ** 2).sum(axis=1) <= r * r] + root
            ok = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
            if np.any(ok):
                x, y = pts[np.argmax(ok)]
                return float(x), float(y)

    def _extend(self, x, y):
        n = len(self.nodes)
        d2 = (self._xy[:n, 0] - x) ** 2 + (self._xy[:n, 1] - y) ** 2
        parent = self.nodes[int(np.argmin(d2))]

        dx, dy = x - parent.x, y - parent.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return None
        if dist > self.extension_range_m:
            scale = self.extension_range_m / dist
            dx, dy, dist = dx * scale, dy * scale, self.extension_range_m
        nx_, ny_ = parent.x + dx, parent.y + dy

        if self.cone_cos is not None and not self._ahead(nx_, ny_):
            return None
        if not self.checker.edge_clear(parent.x, parent.y, nx_, ny_):
            return None

        yaw = math.atan2(dy, dx)
        distance = parent.distance + dist
        view_gain = self.view.gain(self.grid, self.snapshot, nx_, ny_, yaw)
        node = Node(
            nx_,
            ny_,
            yaw,
            parent=parent,
            distance=distance,
            view_gain=view_gain,
            gain=parent.gain + view_gain * math.exp(-self.degressive_coeff * distance),
        )
        self._append(node)
        self.counter += 1
        if node.gain > self.best_gain:
            self.best_gain = node.gain
            self.best_node = node
        return node

    def _ahead(self, x, y):
        """Inside the forward cone about the vehicle's heading, seen from the root."""
        dx, dy = x - self.root.x, y - self.root.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        cos_off = (dx * math.cos(self.root.yaw) + dy * math.sin(self.root.yaw)) / dist
        return cos_off >= self.cone_cos

    def _append(self, node):
        n = len(self.nodes)
        if n == self._xy.shape[0]:
            self._xy = np.concatenate([self._xy, np.zeros_like(self._xy)])
        self._xy[n] = (node.x, node.y)
        self.nodes.append(node)

    # ── readout ───────────────────────────────────────────────────────────
    def best_first_edge(self):
        """Child of the root on the best branch (getBestEdge), or None."""
        node = self.best_node
        if node is None:
            return None
        while node.parent is not self.root:
            node = node.parent
        return node

    def best_branch_states(self):
        """Best branch below its first edge, shallowest first (memorizeBestBranch).

        The first edge is excluded: it is what the vehicle is about to drive,
        and its endpoint becomes the next root.
        """
        states = []
        node = self.best_node
        while node is not None and node.parent is not None and node.parent.parent is not None:
            states.append((node.x, node.y))
            node = node.parent
        states.reverse()
        return states

    def best_branch_nodes(self):
        nodes = []
        node = self.best_node
        while node is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def edges(self):
        return [((n.parent.x, n.parent.y), (n.x, n.y)) for n in self.nodes if n.parent is not None]
