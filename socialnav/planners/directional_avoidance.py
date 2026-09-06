"""Direction-aware reactive escape control for nearby pedestrians."""

from collections.abc import Iterable
from math import hypot

from .dynamic_avoidance import Position, compute_speed_scale

ESCAPE_SPEED_SCALE = 0.25
DIRECTION_DOT_TOLERANCE = 1e-12


def is_separation_increasing(
    robot_position: Position,
    pedestrian_position: Position,
    intended_motion: Position,
    *,
    tolerance: float = DIRECTION_DOT_TOLERANCE,
) -> bool:
    """Return whether motion points clearly away from the pedestrian."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    to_human = (
        pedestrian_position[0] - robot_position[0],
        pedestrian_position[1] - robot_position[1],
    )
    if hypot(*to_human) == 0.0 or hypot(*intended_motion) == 0.0:
        return False

    direction_dot = (
        intended_motion[0] * to_human[0]
        + intended_motion[1] * to_human[1]
    )
    return direction_dot < -tolerance


def compute_directional_speed_scale(
    robot_position: Position,
    pedestrian_position: Position,
    intended_motion: Position,
    stop_distance: float,
    slow_distance: float,
    escape_speed_scale: float = ESCAPE_SPEED_SCALE,
    *,
    additional_pedestrian_positions: Iterable[Position] = (),
) -> float:
    """Return distance-based speed with a conservative away-motion exception.

    Outside the hard-stop region this exactly delegates to the existing
    distance-only controller, where the closest pedestrian determines the
    distance scale.  At or inside ``stop_distance``, escape speed is allowed
    only when ``dot(intended_motion, to_human)`` is less than
    ``-DIRECTION_DOT_TOLERANCE`` for **every** pedestrian inside the
    hard-stop region; tangential, zero-length, coincident, and toward-human
    cases remain stopped.

    With no additional pedestrians this reproduces the original single-human
    controller exactly.
    """
    if not 0.0 < escape_speed_scale <= 1.0:
        raise ValueError("escape_speed_scale must be in (0, 1]")

    positions = (
        pedestrian_position,
        *tuple(additional_pedestrian_positions),
    )
    distance_scale = min(
        compute_speed_scale(
            robot_position,
            position,
            stop_distance,
            slow_distance,
        )
        for position in positions
    )
    if distance_scale > 0.0:
        return distance_scale

    critical_positions = [
        position
        for position in positions
        if hypot(
            position[0] - robot_position[0],
            position[1] - robot_position[1],
        )
        <= stop_distance
    ]
    if not critical_positions:
        return 0.0

    if hypot(*intended_motion) == 0.0:
        return 0.0
    for position in critical_positions:
        to_human = (
            position[0] - robot_position[0],
            position[1] - robot_position[1],
        )
        if hypot(*to_human) == 0.0:
            return 0.0
        direction_dot = (
            intended_motion[0] * to_human[0]
            + intended_motion[1] * to_human[1]
        )
        if direction_dot >= -DIRECTION_DOT_TOLERANCE:
            return 0.0

    return escape_speed_scale