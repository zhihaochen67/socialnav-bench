"""Direction-aware reactive escape control for a nearby pedestrian."""

from math import hypot

from .dynamic_avoidance import Position, compute_speed_scale

ESCAPE_SPEED_SCALE = 0.25
DIRECTION_DOT_TOLERANCE = 1e-12


def compute_directional_speed_scale(
    robot_position: Position,
    pedestrian_position: Position,
    intended_motion: Position,
    stop_distance: float,
    slow_distance: float,
    escape_speed_scale: float = ESCAPE_SPEED_SCALE,
) -> float:
    """Return distance-based speed with a conservative away-motion exception.

    Outside the hard-stop region this exactly delegates to the existing
    distance-only controller. At or inside ``stop_distance``, escape speed is
    allowed only when ``dot(intended_motion, to_human)`` is less than
    ``-DIRECTION_DOT_TOLERANCE``. Tangential, zero-length, coincident, and
    toward-human cases remain stopped.
    """
    if not 0.0 < escape_speed_scale <= 1.0:
        raise ValueError("escape_speed_scale must be in (0, 1]")

    distance_scale = compute_speed_scale(
        robot_position,
        pedestrian_position,
        stop_distance,
        slow_distance,
    )
    if distance_scale > 0.0:
        return distance_scale

    to_human = (
        pedestrian_position[0] - robot_position[0],
        pedestrian_position[1] - robot_position[1],
    )
    if hypot(*to_human) == 0.0 or hypot(*intended_motion) == 0.0:
        return 0.0

    direction_dot = (
        intended_motion[0] * to_human[0]
        + intended_motion[1] * to_human[1]
    )
    if direction_dot < -DIRECTION_DOT_TOLERANCE:
        return escape_speed_scale
    return 0.0
