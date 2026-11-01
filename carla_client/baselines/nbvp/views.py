"""
View gain -- the 2D stand-in for nbvplanner's information gain (rrt.cpp gain()).

Upstream scores a viewpoint by the volume of unmapped voxels that lie inside
the camera frustum, within `nbvp/gain/range`, and that a ray from the
viewpoint reaches without crossing an occupied voxel (getVisibility with
stop_at_unknown_cell=false: unknown space does not occlude). With the weights
in upstream's exploration.yaml (unmapped=1, free=occupied=area=0) that volume
is the whole gain.

On the BEV grid the same quantity is the *area* of UNKNOWN cells, with the
frustum reduced to two dimensions:

  * horizontal FOV -> a wedge of +-hfov/2 about the node's heading
  * vertical FOV   -> a minimum ground range. A level camera at height h with
                      vertical FOV v first sees the road at h / tan(v/2);
                      nearer cells are below the bottom of the image
  * occlusion      -> each ray stops at its first OCC cell

Occlusion in 2D is conservative: any OCC cell blocks the ray, whereas the real
camera at 1.7 m sees over a low wall. That under-counts gain behind low
obstacles; it never invents gain.

Rays are marched rather than cells enumerated. A fan dense enough that
adjacent rays are half a cell apart at the far range touches every cell in the
wedge, and np.unique removes the duplicates near the apex.
"""

import numpy as np

from baselines.yamauchi.grid_map import OCC, UNKNOWN


class ViewModel(object):
    """Forward camera on a ground vehicle, as seen by the gain function."""

    def __init__(
        self,
        resolution_m,
        hfov_deg=90.0,
        vfov_deg=73.74,
        gain_range_m=12.0,
        cam_height_m=1.7,
        cam_forward_m=1.0,
    ):
        self.res = float(resolution_m)
        self.hfov = np.radians(hfov_deg)
        self.gain_range_m = float(gain_range_m)
        self.cam_forward_m = float(cam_forward_m)
        self.min_ground_range_m = float(cam_height_m) / np.tan(np.radians(vfov_deg) / 2.0)

        step = 0.5 * self.res
        n_rays = int(np.ceil(self.hfov * self.gain_range_m / step)) + 1
        self.rel_bearings = np.linspace(-self.hfov / 2.0, self.hfov / 2.0, n_rays)
        self.radii = np.arange(step, self.gain_range_m + 1e-9, step)
        self.counted = self.radii >= self.min_ground_range_m

    def visible_unknown(self, grid, snapshot, x, y, yaw):
        """Flat indices (ix * ny + iy) of UNKNOWN cells the camera would see.

        (x, y) is the vehicle origin and yaw its heading in radians; the
        camera sits cam_forward_m ahead of the origin.
        """
        cx = x + self.cam_forward_m * np.cos(yaw)
        cy = y + self.cam_forward_m * np.sin(yaw)
        bearings = yaw + self.rel_bearings
        px = cx + np.cos(bearings)[:, None] * self.radii[None, :]
        py = cy + np.sin(bearings)[:, None] * self.radii[None, :]

        ix, iy = grid.world_to_cell(px, py)
        inside = grid.in_bounds(ix, iy)
        ix = np.clip(ix, 0, grid.nx - 1)
        iy = np.clip(iy, 0, grid.ny - 1)
        state = snapshot[ix, iy]

        # Leaving the map ends a ray just as an obstacle does: past the edge
        # there is nothing the vehicle could ever map.
        stop = (state == OCC) | ~inside
        blocked = np.logical_or.accumulate(stop, axis=1)
        seen = ~blocked & (state == UNKNOWN) & self.counted[None, :]
        if not np.any(seen):
            return np.empty(0, dtype=np.int64)
        return np.unique(ix[seen] * grid.ny + iy[seen])

    def gain(self, grid, snapshot, x, y, yaw):
        """Unknown area (m^2) seen from pose (x, y, yaw) -- upstream's gain * disc^3."""
        return float(self.visible_unknown(grid, snapshot, x, y, yaw).size) * self.res * self.res
