"""Distance-based social cost for a pedestrian's personal space."""

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
