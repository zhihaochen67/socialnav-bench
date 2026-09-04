"""Pure episode-level navigation metrics."""

from collections.abc import Sequence
from math import hypot

Position = tuple[float, float]


def compute_path_length(trajectory: Sequence[Position]) -> float:
    """Return the total Euclidean length of a 2D trajectory."""
    return sum(
        hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(trajectory, trajectory[1:])
    )


def is_success(
    final_position: Position,
    goal_position: Position,
    tolerance: float,
) -> bool:
    """Return whether the final position is within the goal tolerance."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    return hypot(
        final_position[0] - goal_position[0],
        final_position[1] - goal_position[1],
    ) <= tolerance


def compute_time_to_goal(
    steps: int,
    dt: float,
    success: bool,
) -> float | None:
    """Return elapsed time for a successful episode, otherwise ``None``."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    if not success:
        return None
    return steps * dt


def compute_spl(
    success: bool,
    shortest_path_length: float,
    actual_path_length: float,
) -> float:
    """Return Success weighted by Path Length for one episode."""
    if shortest_path_length < 0.0:
        raise ValueError("shortest_path_length must be non-negative")
    if actual_path_length < 0.0:
        raise ValueError("actual_path_length must be non-negative")

    if not success:
        return 0.0
    if shortest_path_length == 0.0:
        return 1.0 if actual_path_length == 0.0 else 0.0

    return shortest_path_length / max(
        actual_path_length,
        shortest_path_length,
    )
