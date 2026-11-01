"""Vanilla Yamauchi (1997) nearest-frontier exploration baseline for CARLA.

A geometric frontier detector over a BEV occupancy grid, with nearest-frontier
goal selection. Pure numpy/scipy, so it runs in-process in the CARLA env, with no
ZMQ server.
"""

from .explorer import ReplanResult, YamauchiExplorer
from .frontier import Blacklist, FrontierCluster, detect, frontier_mask
from .grid_map import FREE, OCC, UNKNOWN, GridMap, intrinsics_from_fov

__all__ = [
    "YamauchiExplorer",
    "ReplanResult",
    "GridMap",
    "intrinsics_from_fov",
    "FREE",
    "OCC",
    "UNKNOWN",
    "detect",
    "frontier_mask",
    "FrontierCluster",
    "Blacklist",
]
