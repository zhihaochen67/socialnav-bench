"""Pure, reproducible failure diagnostics for benchmark episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, hypot

from socialnav.env.world import STOP_DISTANCE
from socialnav.evaluation import EpisodeResult
from socialnav.metrics import Position

from .scenario import Scenario
from .space_time_diagnostics import SpaceTimePlanningCall

STOPPED_SPEED_TOLERANCE = 1e-12
PATH_NEAR_THRESHOLD = STOP_DISTANCE
LATE_EPISODE_FRACTION = 0.25
LATE_STOPPED_FRACTION_THRESHOLD = 0.50
LONGEST_STOPPED_FRACTION_THRESHOLD = 0.25


@dataclass(frozen=True)
class ShieldCollisionAttribution:
    """Shield evidence aligned to one pedestrian''s first collision."""

    pedestrian_index: int
    execution_phase: str
    shield_evaluated_interval: bool
    candidate_action_selected: str | None
    predicted_minimum_separation: float | None
    actual_first_collision_time: float
    pedestrian_moved_into_robot: bool


@dataclass(frozen=True)
class EpisodeTrace:
    """Runner state needed to diagnose an episode without changing metrics."""

    planned_path: tuple[Position, ...]
    speed_scales: tuple[float, ...]
    timed_out: bool
    final_robot_position: Position
    final_pedestrian_position: Position
    max_steps: int
    replan_count: int = 0
    replan_steps: tuple[int, ...] = ()
    successful_replans: int = 0
    failed_replans: int = 0
    recovery_count: int = 0
    recovery_trigger_steps: tuple[int, ...] = ()
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_path_lengths: tuple[int, ...] = ()
    planned_wait_actions: int = 0
    executed_wait_actions: int = 0
    planned_move_actions: int = 0
    spacetime_plan_count: int = 0
    spacetime_planning_failures: int = 0
    total_intentional_wait_steps: int = 0
    reactive_stopped_steps: int = 0
    space_time_planning_calls: tuple[SpaceTimePlanningCall, ...] = ()
    continuous_bridge_attempts: int = 0
    continuous_bridge_successes: int = 0
    continuous_bridge_failures: int = 0
    bridge_target_cells: tuple[tuple[int, int], ...] = ()
    bridge_distances: tuple[float, ...] = ()
    bridge_min_predicted_separations: tuple[float, ...] = ()
    collision_egress_attempts: int = 0
    collision_egress_successes: int = 0
    collision_egress_failures: int = 0
    progress_stall_events: int = 0
    exact_zero_stall_events: int = 0
    suppressed_duplicate_replans: int = 0
    robust_replan_count: int = 0
    robust_replan_successes: int = 0
    robust_replan_failures: int = 0
    robust_planning_failure_reasons: tuple[str, ...] = ()
    robust_episode_failure_reason: str | None = None
    pedestrian_count: int = 1
    initial_pedestrian_positions: tuple[Position, ...] = ()
    initial_pedestrian_velocities: tuple[Position, ...] = ()
    pedestrian_targets: tuple[Position, ...] = ()
    final_pedestrian_positions: tuple[Position, ...] = ()
    per_human_minimum_distances: tuple[float | None, ...] = ()
    collision_human_indices: tuple[int, ...] = ()
    blocking_human_indices: tuple[int, ...] = ()
    minimum_predicted_separation: float | None = None
    shield_checks: int = 0
    shield_activations: int = 0
    shield_safe_passthroughs: int = 0
    unsafe_planned_moves: int = 0
    unsafe_waits: int = 0
    local_override_count: int = 0
    local_override_actions: tuple[str, ...] = ()
    local_override_target_cells: tuple[tuple[int, int], ...] = ()
    local_override_unsafe_human_indices: tuple[tuple[int, ...], ...] = ()
    candidate_actions_evaluated: int = 0
    candidate_actions_safe: int = 0
    shield_trigger_reasons: tuple[str, ...] = ()
    shield_min_predicted_separation: float | None = None
    post_override_replans: int = 0
    post_override_replan_successes: int = 0
    post_override_replan_failures: int = 0
    no_safe_local_action_events: int = 0
    shield_collision_attributions: tuple[
        ShieldCollisionAttribution, ...
    ] = ()


@dataclass(frozen=True)
class FailureDiagnostic:
    """Structured evidence and deterministic classification for one failure."""

    scenario_id: str
    method: str
    success: bool
    timed_out: bool
    steps: int
    max_steps: int
    final_robot_position: Position
    final_pedestrian_position: Position
    final_distance_to_goal: float
    minimum_human_distance: float | None
    social_violation_rate: float
    human_collision: bool
    obstacle_collision: bool
    planned_path: tuple[Position, ...]
    pedestrian_target: Position
    pedestrian_target_on_path: bool
    pedestrian_target_distance_to_path: float
    pedestrian_final_position_on_or_near_path: bool
    pedestrian_final_distance_to_path: float
    nearest_path_waypoint: Position
    final_robot_to_pedestrian_distance: float
    robot_stopped_steps: int
    robot_stopped_fraction: float
    late_robot_stopped_fraction: float
    longest_consecutive_stop_steps: int
    likely_failure_reason: str
    replan_count: int
    replan_steps: tuple[int, ...]
    successful_replans: int
    failed_replans: int
    recovery_count: int
    recovery_trigger_steps: tuple[int, ...]
    successful_recoveries: int
    failed_recoveries: int
    recovery_path_lengths: tuple[int, ...]
    planned_wait_actions: int
    executed_wait_actions: int
    planned_move_actions: int
    spacetime_plan_count: int
    spacetime_planning_failures: int
    total_intentional_wait_steps: int
    reactive_stopped_steps: int


def _point_to_segment_distance(
    point: Position,
    start: Position,
    end: Position,
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    squared_length = delta_x * delta_x + delta_y * delta_y
    if squared_length == 0.0:
        return hypot(point[0] - start[0], point[1] - start[1])

    projection = (
        (point[0] - start[0]) * delta_x
        + (point[1] - start[1]) * delta_y
    ) / squared_length
    projection = min(max(projection, 0.0), 1.0)
    nearest_x = start[0] + projection * delta_x
    nearest_y = start[1] + projection * delta_y
    return hypot(point[0] - nearest_x, point[1] - nearest_y)


def point_to_route_distance(
    point: Position,
    route: Sequence[Position],
) -> float:
    """Return minimum world-space distance from a point to a polyline route."""
    if not route:
        raise ValueError("route must not be empty")
    if len(route) == 1:
        return hypot(point[0] - route[0][0], point[1] - route[0][1])
    return min(
        _point_to_segment_distance(point, start, end)
        for start, end in zip(route, route[1:])
    )


def nearest_route_waypoint(
    point: Position,
    route: Sequence[Position],
) -> Position:
    """Return the closest route waypoint to a world-space point."""
    if not route:
        raise ValueError("route must not be empty")
    return min(
        route,
        key=lambda waypoint: hypot(
            point[0] - waypoint[0],
            point[1] - waypoint[1],
        ),
    )


def is_point_on_or_near_route(
    point: Position,
    route: Sequence[Position],
    *,
    threshold: float = PATH_NEAR_THRESHOLD,
) -> bool:
    """Return whether a point is within ``threshold`` metres of a route."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    return point_to_route_distance(point, route) <= threshold


def _is_stopped(
    speed_scale: float,
    tolerance: float = STOPPED_SPEED_TOLERANCE,
) -> bool:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    return abs(speed_scale) <= tolerance


def count_stopped_steps(
    speed_scales: Sequence[float],
    *,
    tolerance: float = STOPPED_SPEED_TOLERANCE,
) -> int:
    """Count speed scales that are zero within a tiny numeric tolerance."""
    return sum(_is_stopped(scale, tolerance) for scale in speed_scales)


def stopped_fraction(
    speed_scales: Sequence[float],
    *,
    tolerance: float = STOPPED_SPEED_TOLERANCE,
) -> float:
    """Return the fraction of recorded control steps spent stopped."""
    if not speed_scales:
        return 0.0
    return count_stopped_steps(
        speed_scales,
        tolerance=tolerance,
    ) / len(speed_scales)


def longest_stopped_streak(
    speed_scales: Sequence[float],
    *,
    tolerance: float = STOPPED_SPEED_TOLERANCE,
) -> int:
    """Return the longest consecutive run of stopped control steps."""
    longest = 0
    current = 0
    for scale in speed_scales:
        if _is_stopped(scale, tolerance):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def late_stopped_fraction(
    speed_scales: Sequence[float],
    *,
    late_fraction: float = LATE_EPISODE_FRACTION,
    tolerance: float = STOPPED_SPEED_TOLERANCE,
) -> float:
    """Return stopped fraction within the final fraction of control steps."""
    if not 0.0 < late_fraction <= 1.0:
        raise ValueError("late_fraction must be in (0, 1]")
    if not speed_scales:
        return 0.0
    late_steps = max(1, ceil(len(speed_scales) * late_fraction))
    return stopped_fraction(
        speed_scales[-late_steps:],
        tolerance=tolerance,
    )


def did_episode_time_out(
    *,
    steps: int,
    max_steps: int,
    path_completed: bool,
) -> bool:
    """Return whether the step budget ended an incomplete route traversal."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return steps >= max_steps and not path_completed


def classify_failure(
    *,
    success: bool,
    timed_out: bool,
    human_collision: bool,
    obstacle_collision: bool,
    pedestrian_near_path: bool,
    robot_to_pedestrian_distance: float,
    late_stop_fraction: float,
    longest_stop_steps: int,
    total_steps: int,
) -> str | None:
    """Classify an episode using explicit, deterministic evidence rules."""
    if success:
        return None
    if human_collision or obstacle_collision:
        return "collision"

    long_stop_threshold = ceil(
        total_steps * LONGEST_STOPPED_FRACTION_THRESHOLD
    )
    has_wait_pattern = (
        late_stop_fraction >= LATE_STOPPED_FRACTION_THRESHOLD
        and longest_stop_steps >= long_stop_threshold
    )
    pedestrian_is_blocking = (
        pedestrian_near_path
        and robot_to_pedestrian_distance
        <= STOP_DISTANCE + STOPPED_SPEED_TOLERANCE
    )

    if timed_out and has_wait_pattern and pedestrian_is_blocking:
        return "pedestrian_blocking_path"
    if timed_out and has_wait_pattern:
        return "reactive_wait_timeout"
    if not timed_out:
        return "goal_not_reached"
    return "other"


def diagnose_failure(
    scenario: Scenario,
    method: str,
    result: EpisodeResult,
    trace: EpisodeTrace,
) -> FailureDiagnostic | None:
    """Build a diagnostic for a failed episode, or ``None`` for success."""
    if result.success:
        return None
    if len(trace.speed_scales) != result.steps:
        raise ValueError("trace must contain one speed scale per episode step")

    goal = (
        scenario.goal[0] * scenario.grid_scale,
        scenario.goal[1] * scenario.grid_scale,
    )
    final_path_distance = point_to_route_distance(
        trace.final_pedestrian_position,
        trace.planned_path,
    )
    target_path_distance = point_to_route_distance(
        scenario.pedestrian_target,
        trace.planned_path,
    )
    robot_to_pedestrian_distance = hypot(
        trace.final_robot_position[0] - trace.final_pedestrian_position[0],
        trace.final_robot_position[1] - trace.final_pedestrian_position[1],
    )
    stopped_steps = count_stopped_steps(trace.speed_scales)
    stopped_step_fraction = stopped_fraction(trace.speed_scales)
    late_stop_fraction = late_stopped_fraction(trace.speed_scales)
    longest_stop_steps = longest_stopped_streak(trace.speed_scales)
    final_position_near_path = (
        final_path_distance <= PATH_NEAR_THRESHOLD
    )
    reason = classify_failure(
        success=result.success,
        timed_out=trace.timed_out,
        human_collision=result.human_collision,
        obstacle_collision=result.obstacle_collision,
        pedestrian_near_path=final_position_near_path,
        robot_to_pedestrian_distance=robot_to_pedestrian_distance,
        late_stop_fraction=late_stop_fraction,
        longest_stop_steps=longest_stop_steps,
        total_steps=result.steps,
    )
    if reason is None:
        raise RuntimeError("failed episode did not receive a classification")

    return FailureDiagnostic(
        scenario_id=scenario.scenario_id,
        method=method,
        success=result.success,
        timed_out=trace.timed_out,
        steps=result.steps,
        max_steps=trace.max_steps,
        final_robot_position=trace.final_robot_position,
        final_pedestrian_position=trace.final_pedestrian_position,
        final_distance_to_goal=hypot(
            trace.final_robot_position[0] - goal[0],
            trace.final_robot_position[1] - goal[1],
        ),
        minimum_human_distance=result.minimum_human_distance,
        social_violation_rate=result.social_violation_rate,
        human_collision=result.human_collision,
        obstacle_collision=result.obstacle_collision,
        planned_path=trace.planned_path,
        pedestrian_target=scenario.pedestrian_target,
        pedestrian_target_on_path=(
            target_path_distance <= PATH_NEAR_THRESHOLD
        ),
        pedestrian_target_distance_to_path=target_path_distance,
        pedestrian_final_position_on_or_near_path=(
            final_position_near_path
        ),
        pedestrian_final_distance_to_path=final_path_distance,
        nearest_path_waypoint=nearest_route_waypoint(
            trace.final_pedestrian_position,
            trace.planned_path,
        ),
        final_robot_to_pedestrian_distance=(
            robot_to_pedestrian_distance
        ),
        robot_stopped_steps=stopped_steps,
        robot_stopped_fraction=stopped_step_fraction,
        late_robot_stopped_fraction=late_stop_fraction,
        longest_consecutive_stop_steps=longest_stop_steps,
        likely_failure_reason=reason,
        replan_count=trace.replan_count,
        replan_steps=trace.replan_steps,
        successful_replans=trace.successful_replans,
        failed_replans=trace.failed_replans,
        recovery_count=trace.recovery_count,
        recovery_trigger_steps=trace.recovery_trigger_steps,
        successful_recoveries=trace.successful_recoveries,
        failed_recoveries=trace.failed_recoveries,
        recovery_path_lengths=trace.recovery_path_lengths,
        planned_wait_actions=trace.planned_wait_actions,
        executed_wait_actions=trace.executed_wait_actions,
        planned_move_actions=trace.planned_move_actions,
        spacetime_plan_count=trace.spacetime_plan_count,
        spacetime_planning_failures=trace.spacetime_planning_failures,
        total_intentional_wait_steps=trace.total_intentional_wait_steps,
        reactive_stopped_steps=trace.reactive_stopped_steps,
    )
