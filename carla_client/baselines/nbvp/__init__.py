"""Receding-horizon next-best-view planner (Bircher et al., ICRA 2016) for CARLA.

Port of github.com/ethz-asl/nbvplanner onto the Yamauchi baseline's BEV
occupancy grid: same map, same sensor, same clearance rule, different choice of
where to drive.
"""

from .collision import EdgeChecker
from .explorer import NBVPExplorer, ReplanResult, segment_waypoints
from .tree import Node, RrtTree
from .views import ViewModel

__all__ = [
    "NBVPExplorer",
    "ReplanResult",
    "segment_waypoints",
    "EdgeChecker",
    "RrtTree",
    "Node",
    "ViewModel",
]
