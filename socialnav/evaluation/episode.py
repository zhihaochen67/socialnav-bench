"""Pure evaluation of one recorded navigation episode."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

from socialnav.metrics import (
    Position,
    compute_minimum_human_distance,
    compute_path_length,
    compute_social_violation_rate,
    compute_spl,
    compute_time_to_goal,
    has_human_collision,
    is_success,
)


@dataclass(frozen=True)
class EpisodeResult:
    """Metrics computed from one completed or terminated episode."""

    success: bool
    path_length: float
    time_to_goal: float | None
    spl: float
    minimum_human_distance: float | None
    social_violation_rate: float
    human_collision: bool
    obstacle_collision: bool
    steps: int


def has_static_obstacle_collision(
    robot_trajectory: Sequence[Position],
    obstacle_positions: Sequence[Position],
    obstacle_half_extent: float,
    robot_radius: float,
) -> bool:
    """Check a circular robot against static axis-aligned square obstacles."""
    if obstacle_half_extent < 0.0:
        raise ValueError("obstacle_half_extent must be non-negative")
    if robot_radius < 0.0:
        raise ValueError("robot_radius must be non-negative")

    for robot_x, robot_y in robot_trajectory:
        for obstacle_x, obstacle_y in obstacle_positions:
            delta_x = max(abs(robot_x - obstacle_x) - obstacle_half_extent, 0.0)
            delta_y = max(abs(robot_y - obstacle_y) - obstacle_half_extent, 0.0)
            if hypot(delta_x, delta_y) <= robot_radius:
                return True

    return False


def evaluate_episode(
    robot_trajectory: Sequence[Position],
    human_trajectories: Sequence[Sequence[Position]],
    *,
    goal_position: Position,
    goal_tolerance: float,
    steps: int,
    dt: float,
    shortest_path_length: float,
    social_distance: float,
    human_collision_distance: float,
    obstacle_positions: Sequence[Position] = (),
    obstacle_half_extent: float = 0.0,
    robot_radius: float = 0.0,
) -> EpisodeResult:
    """Compute all metrics for one timestep-aligned recorded episode."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not robot_trajectory:
        raise ValueError("robot_trajectory must include the initial state")
    if len(robot_trajectory) != steps + 1:
        raise ValueError(
            "robot_trajectory must contain the initial state plus one "
            "sample per simulation step"
        )

    success = is_success(
        robot_trajectory[-1],
        goal_position,
        goal_tolerance,
    )
    path_length = compute_path_length(robot_trajectory)

    return EpisodeResult(
        success=success,
        path_length=path_length,
        time_to_goal=compute_time_to_goal(steps, dt, success),
        spl=compute_spl(success, shortest_path_length, path_length),
        minimum_human_distance=compute_minimum_human_distance(
            robot_trajectory,
            human_trajectories,
        ),
        social_violation_rate=compute_social_violation_rate(
            robot_trajectory,
            human_trajectories,
            social_distance,
        ),
        human_collision=has_human_collision(
            robot_trajectory,
            human_trajectories,
            human_collision_distance,
        ),
        obstacle_collision=has_static_obstacle_collision(
            robot_trajectory,
            obstacle_positions,
            obstacle_half_extent,
            robot_radius,
        ),
        steps=steps,
    )
