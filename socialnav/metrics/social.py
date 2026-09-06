"""Pure episode-level metrics for robot-human interaction."""

from collections.abc import Sequence
from math import hypot

from .navigation import Position


def _validate_aligned_trajectories(
    robot_trajectory: Sequence[Position],
    human_trajectories: Sequence[Sequence[Position]],
) -> None:
    expected_length = len(robot_trajectory)
    for index, human_trajectory in enumerate(human_trajectories):
        if len(human_trajectory) != expected_length:
            raise ValueError(
                f"human trajectory {index} must have length "
                f"{expected_length} to match the robot trajectory"
            )


def _distance(first: Position, second: Position) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def compute_minimum_human_distance(
    robot_trajectory: Sequence[Position],
    human_trajectories: Sequence[Sequence[Position]],
) -> float | None:
    """Return the closest synchronized robot-human distance in an episode."""
    if not human_trajectories:
        return None

    _validate_aligned_trajectories(robot_trajectory, human_trajectories)
    if not robot_trajectory:
        return None

    return min(
        _distance(robot_position, human_trajectory[timestep])
        for timestep, robot_position in enumerate(robot_trajectory)
        for human_trajectory in human_trajectories
    )


def compute_social_violation_rate(
    robot_trajectory: Sequence[Position],
    human_trajectories: Sequence[Sequence[Position]],
    social_distance: float,
) -> float:
    """Return the fraction of timesteps inside any human's social radius."""
    if social_distance <= 0.0:
        raise ValueError("social_distance must be positive")
    if not human_trajectories:
        return 0.0

    _validate_aligned_trajectories(robot_trajectory, human_trajectories)
    if not robot_trajectory:
        return 0.0

    violating_timesteps = sum(
        any(
            _distance(robot_position, human_trajectory[timestep])
            < social_distance
            for human_trajectory in human_trajectories
        )
        for timestep, robot_position in enumerate(robot_trajectory)
    )
    return violating_timesteps / len(robot_trajectory)


def has_human_collision(
    robot_trajectory: Sequence[Position],
    human_trajectories: Sequence[Sequence[Position]],
    collision_distance: float,
) -> bool:
    """Return whether any synchronized robot-human pair collides."""
    if collision_distance < 0.0:
        raise ValueError("collision_distance must be non-negative")
    if not human_trajectories:
        return False

    _validate_aligned_trajectories(robot_trajectory, human_trajectories)
    return any(
        _distance(robot_position, human_trajectory[timestep])
        <= collision_distance
        for timestep, robot_position in enumerate(robot_trajectory)
        for human_trajectory in human_trajectories
    )

def compute_per_human_minimum_distances(
    robot_trajectory: Sequence[Position],
    human_trajectories: Sequence[Sequence[Position]],
) -> tuple[float | None, ...]:
    """Return the closest synchronized distance for each pedestrian."""
    _validate_aligned_trajectories(robot_trajectory, human_trajectories)
    if not robot_trajectory:
        return tuple(None for _ in human_trajectories)

    return tuple(
        min(
            _distance(robot_position, human_trajectory[timestep])
            for timestep, robot_position in enumerate(robot_trajectory)
        )
        for human_trajectory in human_trajectories
    )


def compute_per_human_social_violation_rates(
    robot_trajectory: Sequence[Position],
    human_trajectories: Sequence[Sequence[Position]],
    social_distance: float,
) -> tuple[float, ...]:
    """Return each pedestrian's fraction of inside-social-radius timesteps."""
    if social_distance <= 0.0:
        raise ValueError("social_distance must be positive")

    _validate_aligned_trajectories(robot_trajectory, human_trajectories)
    if not robot_trajectory:
        return tuple(0.0 for _ in human_trajectories)

    return tuple(
        sum(
            _distance(robot_position, human_trajectory[timestep])
            < social_distance
            for timestep, robot_position in enumerate(robot_trajectory)
        )
        / len(robot_trajectory)
        for human_trajectory in human_trajectories
    )


def colliding_human_indices(
    robot_trajectory: Sequence[Position],
    human_trajectories: Sequence[Sequence[Position]],
    collision_distance: float,
) -> tuple[int, ...]:
    """Return the index of every pedestrian involved in any collision."""
    if collision_distance < 0.0:
        raise ValueError("collision_distance must be non-negative")

    _validate_aligned_trajectories(robot_trajectory, human_trajectories)
    return tuple(
        index
        for index, human_trajectory in enumerate(human_trajectories)
        if any(
            _distance(robot_position, human_trajectory[timestep])
            <= collision_distance
            for timestep, robot_position in enumerate(robot_trajectory)
        )
    )