"""Path-planning algorithms."""

from .astar import astar
from .dynamic_avoidance import compute_speed_scale

__all__ = ["astar", "compute_speed_scale"]
