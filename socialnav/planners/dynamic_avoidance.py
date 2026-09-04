"""Deterministic reactive speed control for a nearby pedestrian."""

from math import hypot

Position = tuple[float, float]


def compute_speed_scale(
    robot_position: Position,
    pedestrian_position: Position,
    stop_distance: float,
    slow_distance: float,
) -> float:
    """Return a speed scale based on the Euclidean separation.

    The robot stops at or inside ``stop_distance``, moves at full speed at or
    beyond ``slow_distance``, and scales linearly between the two thresholds.
    """
    if stop_distance < 0.0:
        raise ValueError("stop_distance must be non-negative")
    if stop_distance >= slow_distance:
        raise ValueError("stop_distance must be less than slow_distance")

    distance = hypot(
        robot_position[0] - pedestrian_position[0],
        robot_position[1] - pedestrian_position[1],
    )
    if distance <= stop_distance:
        return 0.0
    if distance >= slow_distance:
        return 1.0

    return (distance - stop_distance) / (slow_distance - stop_distance)
