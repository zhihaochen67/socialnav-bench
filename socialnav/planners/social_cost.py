"""Distance-based social cost for a pedestrian's personal space."""

from collections.abc import Iterable
from math import hypot

Position = tuple[float, float]


def compute_social_cost(
    robot_position: Position,
    pedestrian_position: Position,
    social_distance: float,
    weight: float,
) -> float:
    """Return the quadratic cost of entering a pedestrian's personal space."""
    if social_distance <= 0.0:
        raise ValueError("social_distance must be positive")
    if weight < 0.0:
        raise ValueError("weight must be non-negative")

    distance = hypot(
        robot_position[0] - pedestrian_position[0],
        robot_position[1] - pedestrian_position[1],
    )
    if distance >= social_distance:
        return 0.0

    return weight * (social_distance - distance) ** 2

def compute_multi_social_cost(
    robot_position: Position,
    pedestrian_positions: Iterable[Position],
    social_distance: float,
    weight: float,
) -> float:
    """Return the unnormalized sum of individual social costs.

    ``N=0`` yields zero and ``N=1`` reproduces :func:`compute_social_cost`
    exactly; the sum is never normalized by pedestrian count.
    """
    if social_distance <= 0.0:
        raise ValueError("social_distance must be positive")
    if weight < 0.0:
        raise ValueError("weight must be non-negative")

    return sum(
        compute_social_cost(
            robot_position,
            pedestrian_position,
            social_distance,
            weight,
        )
        for pedestrian_position in pedestrian_positions
    )