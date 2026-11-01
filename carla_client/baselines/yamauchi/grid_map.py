"""
Persistent global 2D BEV occupancy grid, built from a forward depth camera.

Coordinates
-----------
Everything here is in CARLA's native world frame (left-handed, Z up). Nothing in
this package cares about handedness as long as it is used consistently. Camera-local axes follow CARLA's sensor convention:
+X forward, +Y right, +Z up.

Pose source is CARLA ground truth (carla.Image.transform, i.e. the sensor's
transform at capture time). No SLAM, no odometry estimation -- this baseline
assumes perfect localization.

Intrinsics
----------
envs/carla/v0.py builds camera_intrinsic with
    fx = W / (2*tan(fov/2))   fy = H / (2*tan(fov/2))
which yields fx=160, fy=120 at 320x240 / 90deg. That is wrong: CARLA's `fov`
attribute is the *horizontal* FOV and CARLA cameras have square pixels, so
fy must equal fx. CARLA's own examples say so explicitly --
examples/lidar_to_camera.py:120-125, also draw_skeleton.py and
client_bounding_boxes.py:

    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    # In this case Fx and Fy are the same since the pixel aspect ratio is 1
    K[0, 0] = K[1, 1] = focal

320x240 is a rectangular *image* made of *square pixels*; the aspect ratio is
carried by cx/cy and the image dimensions, not by the focal lengths. The true
vertical FOV is 2*atan(120/160) = 73.7 deg. Using fy=120 stretches
backprojected points vertically by 1.33x, which bows the ground plane and
corrupts the height-band classification this module depends on.

So we compute K locally (intrinsics_from_fov) rather than reading
base_env.camera_intrinsic. Deliberately not fixing v0.py here: other baselines
have already produced runs against the existing value, and silently changing
it would invalidate them.

Depth-to-scan reduction
-----------------------
Rather than ray-casting every pixel, backprojected points are binned by
horizontal bearing into an angular scan (the same reduction ROS's
depthimage_to_laserscan does). Per bearing bin we keep the nearest
obstacle-height return and the farthest ground return. That turns ~19k rays
per frame into a few hundred, which keeps integration cheap enough to run
every simulator step in pure numpy.
"""

import numpy as np
from scipy import ndimage

UNKNOWN = -1
FREE = 0
OCC = 1


def intrinsics_from_fov(width, height, hfov_deg):
    """Pinhole K for a CARLA camera: square pixels, `fov` is horizontal."""
    focal = width / (2.0 * np.tan(hfov_deg * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


class GridMap:
    """Log-odds 2D occupancy grid over a fixed world-frame bounding box."""

    def __init__(
        self,
        min_x,
        min_y,
        max_x,
        max_y,
        resolution_m=0.5,
        l_free=0.4,
        l_occ=0.85,
        l_clamp=4.0,
        free_thresh=-0.2,
        occ_thresh=0.4,
        min_obstacle_cells=4,
    ):
        self.res = float(resolution_m)
        self.origin_x = float(min_x)
        self.origin_y = float(min_y)
        self.nx = int(np.ceil((max_x - min_x) / self.res))
        self.ny = int(np.ceil((max_y - min_y) / self.res))
        self.log_odds = np.zeros((self.nx, self.ny), dtype=np.float32)

        self.l_free = float(l_free)
        self.l_occ = float(l_occ)
        self.l_clamp = float(l_clamp)
        self.free_thresh = float(free_thresh)
        self.occ_thresh = float(occ_thresh)
        self.min_obstacle_cells = int(min_obstacle_cells)

    @classmethod
    def from_carla_map(cls, carla_map, resolution_m=0.5, margin_m=100.0, **kwargs):
        """Size the grid from the town's spawn-point extent, plus margin.

        Spawn points do not cover every drivable metre, hence the generous
        default margin -- the grid is cheap (a 2km town at 0.5m is ~64MB of
        float32) and running off the edge silently loses map, so err large.
        """
        pts = carla_map.get_spawn_points()
        xs = np.array([p.location.x for p in pts], dtype=np.float64)
        ys = np.array([p.location.y for p in pts], dtype=np.float64)
        return cls(
            min_x=float(xs.min()) - margin_m,
            min_y=float(ys.min()) - margin_m,
            max_x=float(xs.max()) + margin_m,
            max_y=float(ys.max()) + margin_m,
            resolution_m=resolution_m,
            **kwargs
        )

    # ── world <-> cell ────────────────────────────────────────────────────
    def world_to_cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin_x) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin_y) / self.res).astype(np.int64)
        return ix, iy

    def cell_to_world(self, ix, iy):
        x = self.origin_x + (np.asarray(ix) + 0.5) * self.res
        y = self.origin_y + (np.asarray(iy) + 0.5) * self.res
        return x, y

    def in_bounds(self, ix, iy):
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    # ── integration ───────────────────────────────────────────────────────
    def integrate(
        self,
        depth,
        cam_transform,
        ground_z,
        hfov_deg=90.0,
        max_range_m=40.0,
        min_range_m=1.0,
        obstacle_band_m=(0.5, 3.0),
        pixel_stride=2,
        n_bearing_bins=256,
    ):
        """Fuse one depth frame.

        depth         : (H, W) float32, metric planar depth from CARLA's depth sensor
        cam_transform : carla.Transform of the sensor at capture time (image.transform)
        ground_z      : world z taken as road level under the vehicle
        """
        H, W = depth.shape
        d = depth[::pixel_stride, ::pixel_stride]
        K = intrinsics_from_fov(W, H, hfov_deg)
        f, cx, cy = K[0, 0], K[0, 2], K[1, 2]

        vs, us = np.mgrid[0:H:pixel_stride, 0:W:pixel_stride]
        d = d.ravel()
        us = us.ravel().astype(np.float64)
        vs = vs.ravel().astype(np.float64)

        # Drop sky and anything past our working range. CARLA's depth sensor
        # saturates at 1000m, so without this the horizon paints obstacles.
        keep = (d > min_range_m) & (d < max_range_m)
        if not np.any(keep):
            return
        d, us, vs = d[keep], us[keep], vs[keep]

        # CARLA sensor-local axes from CV pixel coords: +X fwd, +Y right, +Z up.
        x_cam = d
        y_cam = (us - cx) * d / f
        z_cam = -(vs - cy) * d / f

        M = np.asarray(cam_transform.get_matrix(), dtype=np.float64)
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=0)
        pts_w = M.dot(pts_cam)
        wx, wy, wz = pts_w[0], pts_w[1], pts_w[2]

        sx = float(M[0, 3])
        sy = float(M[1, 3])

        dx = wx - sx
        dy = wy - sy
        horiz = np.hypot(dx, dy)
        valid = horiz > 1e-6
        if not np.any(valid):
            return
        dx, dy, horiz, wz = dx[valid], dy[valid], horiz[valid], wz[valid]

        bearing = np.arctan2(dy, dx)
        lo, hi = obstacle_band_m
        is_obstacle = (wz >= ground_z + lo) & (wz <= ground_z + hi)
        is_ground = wz < ground_z + lo

        # ── reduce to an angular scan ────────────────────────────────────
        # Bin over the observed bearing span rather than a fixed [-fov/2, fov/2]
        # window, so the binning stays correct under vehicle yaw and pitch.
        b_min, b_max = bearing.min(), bearing.max()
        if b_max - b_min < 1e-9:
            return
        bin_idx = np.clip(
            ((bearing - b_min) / (b_max - b_min) * n_bearing_bins).astype(np.int64),
            0,
            n_bearing_bins - 1,
        )

        obs_range = np.full(n_bearing_bins, np.inf)
        if np.any(is_obstacle):
            np.minimum.at(obs_range, bin_idx[is_obstacle], horiz[is_obstacle])

        ground_range = np.zeros(n_bearing_bins)
        if np.any(is_ground):
            np.maximum.at(ground_range, bin_idx[is_ground], horiz[is_ground])

        bin_centers = b_min + (np.arange(n_bearing_bins) + 0.5) * (b_max - b_min) / n_bearing_bins

        has_obstacle = np.isfinite(obs_range)
        # Free space runs to the obstacle where there is one, otherwise only as
        # far as the farthest ground return -- past that we have observed
        # nothing and the cells must stay UNKNOWN for frontiers to exist.
        free_to = np.where(has_obstacle, obs_range, ground_range)
        active = free_to > self.res
        if not np.any(active):
            return

        # ── ray sampling ─────────────────────────────────────────────────
        step = self.res * 0.5
        n_steps = int(np.ceil(max_range_m / step)) + 1
        t = np.arange(n_steps) * step

        ranges = free_to[active]
        cos_b = np.cos(bin_centers[active])
        sin_b = np.sin(bin_centers[active])

        # Stop half a cell short of the obstacle so the free sweep does not
        # erase the occupied cell it terminates on.
        mask = t[None, :] < (ranges[:, None] - self.res * 0.5)
        fx_w = sx + t[None, :] * cos_b[:, None]
        fy_w = sy + t[None, :] * sin_b[:, None]

        fix, fiy = self.world_to_cell(fx_w[mask], fy_w[mask])
        ok = self.in_bounds(fix, fiy)
        fix, fiy = fix[ok], fiy[ok]

        if fix.size:
            flat = np.unique(fix * self.ny + fiy)
            np.subtract.at(self.log_odds.reshape(-1), flat, self.l_free)

        # ── obstacle endpoints ───────────────────────────────────────────
        hit = active & has_obstacle
        if np.any(hit):
            hr = obs_range[hit]
            hx = sx + hr * np.cos(bin_centers[hit])
            hy = sy + hr * np.sin(bin_centers[hit])
            oix, oiy = self.world_to_cell(hx, hy)
            ok = self.in_bounds(oix, oiy)
            oix, oiy = oix[ok], oiy[ok]
            if oix.size:
                flat = np.unique(oix * self.ny + oiy)
                np.add.at(self.log_odds.reshape(-1), flat, self.l_occ)

        np.clip(self.log_odds, -self.l_clamp, self.l_clamp, out=self.log_odds)

    # ── readout ───────────────────────────────────────────────────────────
    def snapshot(self):
        """Tri-state view of the map: UNKNOWN / FREE / OCC.

        Obstacle blobs smaller than `min_obstacle_cells` are dropped back to
        FREE. Road grade puts distant ground returns above the flat-world
        height band, so a handful of isolated cells get flagged as obstacles
        in the middle of an otherwise clear road. They are a tiny fraction of
        the occupied cells (~8% in Town01) but the planner inflates each one
        into a disk of the vehicle's half-width, and two such disks are enough
        to seal a road and strand the vehicle. Measured on a stuck run:
        dropping single-cell blobs alone took the reachable area around the
        vehicle from 0 back to 464 m^2.
        """
        out = np.full(self.log_odds.shape, UNKNOWN, dtype=np.int8)
        out[self.log_odds < self.free_thresh] = FREE

        occ = self.log_odds > self.occ_thresh
        if self.min_obstacle_cells > 1 and np.any(occ):
            labels, n = ndimage.label(occ, structure=np.ones((3, 3), dtype=np.int8))
            if n:
                sizes = np.bincount(labels.ravel())
                too_small = sizes < self.min_obstacle_cells
                too_small[0] = False
                occ = occ & ~too_small[labels]

        out[occ] = OCC
        return out

    def is_clear(self, x, y, yaw_deg, distance_m, half_width_m=1.2):
        """True if no OCC cell lies in a corridor ahead of (x, y) along yaw.

        Used by the controller's fallback creep. Driving forward blind when no
        frontier is reachable walks straight into walls the map already knows
        about; this is the minimum check that makes the fallback safe. A
        negative distance_m checks behind the vehicle instead.
        """
        yaw = np.radians(yaw_deg)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        step = self.res * 0.5
        n = max(1, int(abs(distance_m) / step))
        sign = 1.0 if distance_m >= 0 else -1.0
        t = sign * np.arange(1, n + 1) * step

        # Three parallel rays cover the vehicle's width.
        offsets = (-half_width_m, 0.0, half_width_m)
        for off in offsets:
            ox = -sin_y * off
            oy = cos_y * off
            px = x + ox + t * cos_y
            py = y + oy + t * sin_y
            ix, iy = self.world_to_cell(px, py)
            ok = self.in_bounds(ix, iy)
            if not np.any(ok):
                continue
            if np.any(self.log_odds[ix[ok], iy[ok]] > self.occ_thresh):
                return False
        return True

    def stats(self):
        snap = self.snapshot()
        total = snap.size
        return {
            "free_cells": int(np.count_nonzero(snap == FREE)),
            "occ_cells": int(np.count_nonzero(snap == OCC)),
            "unknown_cells": int(np.count_nonzero(snap == UNKNOWN)),
            "total_cells": int(total),
            "cell_area_m2": float(self.res ** 2),
        }
