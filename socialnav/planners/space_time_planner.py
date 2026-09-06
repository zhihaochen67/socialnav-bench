"""Deterministic time-expanded social A* with explicit wait actions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import floor, hypot, inf
from typing import Literal

from socialnav.env.demo_map import grid_to_world
from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.planners.pedestrian_prediction import (
    PedestrianPredictionState,
    predict_pedestrian_position_at_time,
)
from socialnav.planners.social_cost import compute_social_cost

State = tuple[int, int, int]
SpaceTimeAction = Literal["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]
SpaceTimeFailureReason = Literal[
    "invalid_start",
    "invalid_goal",
    "start_in_predicted_collision",
    "no_safe_first_action",
    "search_exhausted",
    "time_horizon_exhausted",
    "goal_unreachable_static",
    "other",
]
SpaceTimeActionRejectionReason = Literal[
    "outside_map",
    "obstacle",
    "predicted_collision",
]
Position = tuple[float, float]

_ACTIONS: tuple[tuple[SpaceTimeAction, Coordinate], ...] = (
    ("UP", (0, -1)),
    ("RIGHT", (1, 0)),
    ("DOWN", (0, 1)),
    ("LEFT", (-1, 0)),
    ("WAIT", (0, 0)),
)


def _collect_pedestrian_states(
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    additional_pedestrians: Iterable[PedestrianPredictionState],
) -> tuple[PedestrianPredictionState, ...]:
    states: list[PedestrianPredictionState] = []
    if pedestrian_position is not None:
        states.append(
            (pedestrian_position, pedestrian_velocity, pedestrian_target)
        )
    states.extend(additional_pedestrians)
    return tuple(states)


@dataclass(frozen=True)
class SpaceTimePlan:
    """Immutable spatial, temporal, and action representation of a plan."""

    spatial_path: tuple[Coordinate, ...]
    timed_states: tuple[State, ...]
    actions: tuple[SpaceTimeAction, ...]
    planned_wait_actions: int
    planned_move_actions: int
    move_duration: float
    estimated_duration: float


@dataclass(frozen=True)
class SpaceTimeActionSafety:
    """Deterministic evidence for one possible first action."""

    action: SpaceTimeAction
    destination: Coordinate
    inside_map: bool
    free: bool
    predicted_separation_start: float
    predicted_separation_midpoint: float
    predicted_separation_end: float
    minimum_predicted_separation: float
    rejected_by_collision_constraint: bool
    rejection_reason: SpaceTimeActionRejectionReason | None


@dataclass(frozen=True)
class SpaceTimeSearchStatistics:
    """Counters describing one space-time search without affecting it."""

    expanded_states: int
    generated_states: int
    maximum_time_index_reached: int
    planning_horizon_reached: bool
    open_set_exhausted: bool
    goal_reached: bool
    returned_path_length: int | None
    planned_wait_count: int


@dataclass(frozen=True)
class SpaceTimePlanningResult:
    """Plan plus deterministic evidence explaining success or failure."""

    plan: SpaceTimePlan | None
    failure_reason: SpaceTimeFailureReason | None
    first_action_safety: tuple[SpaceTimeActionSafety, ...]
    statistics: SpaceTimeSearchStatistics


def duration_to_simulation_steps(duration: float, dt: float) -> int:
    """Round duration/dt to nearest integer, with exact ties rounded up."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return max(1, floor(duration / dt + 0.5))


def compute_time_aligned_social_cost(
    coordinate: Coordinate,
    time_index: int,
    *,
    pedestrian_position: Position,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    move_duration: float,
    grid_scale: float,
    social_distance: float,
    social_weight: float,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> float:
    """Evaluate summed social cost at the state's exact estimated arrival time.

    Every pedestrian is predicted independently at the arrival time and the
    individual social costs are summed without normalization.  With no
    additional pedestrians this is the original single-human cost.
    """
    if time_index < 0:
        raise ValueError("time_index must be non-negative")
    arrival_time = time_index * move_duration
    pedestrians = _collect_pedestrian_states(
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        additional_pedestrians,
    )
    return sum(
        compute_social_cost(
            grid_to_world(coordinate, grid_scale),
            predict_pedestrian_position_at_time(
                position,
                velocity,
                arrival_time,
                target=target,
            ),
            social_distance,
            social_weight,
        )
        for position, velocity, target in pedestrians
    )


def is_space_time_action_safe(
    start: Coordinate,
    end: Coordinate,
    start_time_index: int,
    *,
    pedestrian_position: Position,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    move_duration: float,
    grid_scale: float,
    collision_distance: float,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> bool:
    """Check start, midpoint, and end separation for one timed action.

    The action is safe only when the predicted separation exceeds the
    collision distance for **every** pedestrian at all three time-aligned
    samples.
    """
    if start_time_index < 0:
        raise ValueError("start_time_index must be non-negative")
    if move_duration <= 0.0:
        raise ValueError("move_duration must be positive")
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")
    if collision_distance <= 0.0:
        raise ValueError("collision_distance must be positive")

    return all(
        separation > collision_distance
        for separation in _space_time_action_separations(
            start,
            end,
            start_time_index,
            pedestrian_position=pedestrian_position,
            pedestrian_velocity=pedestrian_velocity,
            pedestrian_target=pedestrian_target,
            move_duration=move_duration,
            grid_scale=grid_scale,
            additional_pedestrians=additional_pedestrians,
        )
    )


def inspect_space_time_first_actions(
    grid_map: GridMap,
    start: Coordinate,
    *,
    pedestrian_position: Position,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    move_duration: float,
    grid_scale: float,
    collision_distance: float,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> tuple[SpaceTimeActionSafety, ...]:
    """Describe map validity and timed separation for all five actions.

    The reported separations are the minimum across all pedestrians at each
    of the three time-aligned samples.
    """
    details = []
    for action, (delta_x, delta_y) in _ACTIONS:
        destination = (start[0] + delta_x, start[1] + delta_y)
        inside_map = grid_map.is_inside(destination)
        free = inside_map and grid_map.is_free(destination)
        separations = _space_time_action_separations(
            start,
            destination,
            0,
            pedestrian_position=pedestrian_position,
            pedestrian_velocity=pedestrian_velocity,
            pedestrian_target=pedestrian_target,
            move_duration=move_duration,
            grid_scale=grid_scale,
            additional_pedestrians=additional_pedestrians,
        )
        minimum_separation = min(separations)
        collision_rejection = minimum_separation <= collision_distance
        if not inside_map:
            rejection_reason: SpaceTimeActionRejectionReason | None = (
                "outside_map"
            )
        elif not free:
            rejection_reason = "obstacle"
        elif collision_rejection:
            rejection_reason = "predicted_collision"
        else:
            rejection_reason = None
        details.append(
            SpaceTimeActionSafety(
                action=action,
                destination=destination,
                inside_map=inside_map,
                free=free,
                predicted_separation_start=separations[0],
                predicted_separation_midpoint=separations[1],
                predicted_separation_end=separations[2],
                minimum_predicted_separation=minimum_separation,
                rejected_by_collision_constraint=collision_rejection,
                rejection_reason=rejection_reason,
            )
        )
    return tuple(details)


def _space_time_action_separations(
    start: Coordinate,
    end: Coordinate,
    start_time_index: int,
    *,
    pedestrian_position: Position,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    move_duration: float,
    grid_scale: float,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> tuple[float, float, float]:
    start_world = grid_to_world(start, grid_scale)
    end_world = grid_to_world(end, grid_scale)
    start_time = start_time_index * move_duration
    pedestrians = _collect_pedestrian_states(
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        additional_pedestrians,
    )

    separations_by_fraction: list[list[float]] = [[], [], []]
    for position, velocity, target in pedestrians:
        for fraction_index, fraction in enumerate((0.0, 0.5, 1.0)):
            robot_position = (
                start_world[0]
                + (end_world[0] - start_world[0]) * fraction,
                start_world[1]
                + (end_world[1] - start_world[1]) * fraction,
            )
            pedestrian_at_time = predict_pedestrian_position_at_time(
                position,
                velocity,
                start_time + move_duration * fraction,
                target=target,
            )
            separations_by_fraction[fraction_index].append(
                hypot(
                    robot_position[0] - pedestrian_at_time[0],
                    robot_position[1] - pedestrian_at_time[1],
                )
            )

    return (
        min(separations_by_fraction[0], default=inf),
        min(separations_by_fraction[1], default=inf),
        min(separations_by_fraction[2], default=inf),
    )


def space_time_social_astar(
    grid_map: GridMap,
    start: Coordinate,
    goal: Coordinate,
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    social_distance: float,
    social_weight: float,
    collision_distance: float,
    grid_scale: float,
    robot_speed: float,
    max_time_seconds: float,
    *,
    diagnostic_results: list[SpaceTimePlanningResult] | None = None,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> SpaceTimePlan | None:
    """Plan in (x, y, time_index) using time-aligned pedestrian occupancy."""
    _validate_endpoint(grid_map, start, "start")
    _validate_endpoint(grid_map, goal, "goal")
    result = diagnose_space_time_social_astar(
        grid_map,
        start,
        goal,
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        social_distance,
        social_weight,
        collision_distance,
        grid_scale,
        robot_speed,
        max_time_seconds,
        additional_pedestrians=additional_pedestrians,
    )
    if diagnostic_results is not None:
        diagnostic_results.append(result)
    return result.plan


def diagnose_space_time_social_astar(
    grid_map: GridMap,
    start: Coordinate,
    goal: Coordinate,
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    social_distance: float,
    social_weight: float,
    collision_distance: float,
    grid_scale: float,
    robot_speed: float,
    max_time_seconds: float,
    *,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> SpaceTimePlanningResult:
    """Run the unchanged search and return deterministic diagnostic evidence."""
    if social_distance <= 0.0:
        raise ValueError("social_distance must be positive")
    if social_weight < 0.0:
        raise ValueError("social_weight must be non-negative")
    if collision_distance <= 0.0:
        raise ValueError("collision_distance must be positive")
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")
    if robot_speed <= 0.0:
        raise ValueError("robot_speed must be positive")
    if max_time_seconds < 0.0:
        raise ValueError("max_time_seconds must be non-negative")

    additional = tuple(additional_pedestrians)
    move_duration = grid_scale / robot_speed
    max_time_index = floor(max_time_seconds / move_duration)
    empty_statistics = SpaceTimeSearchStatistics(
        expanded_states=0,
        generated_states=0,
        maximum_time_index_reached=0,
        planning_horizon_reached=False,
        open_set_exhausted=False,
        goal_reached=False,
        returned_path_length=None,
        planned_wait_count=0,
    )
    if not grid_map.is_inside(start) or not grid_map.is_free(start):
        return SpaceTimePlanningResult(
            plan=None,
            failure_reason="invalid_start",
            first_action_safety=(),
            statistics=empty_statistics,
        )
    if not grid_map.is_inside(goal) or not grid_map.is_free(goal):
        return SpaceTimePlanningResult(
            plan=None,
            failure_reason="invalid_goal",
            first_action_safety=(),
            statistics=empty_statistics,
        )

    first_action_safety = inspect_space_time_first_actions(
        grid_map,
        start,
        pedestrian_position=pedestrian_position,
        pedestrian_velocity=pedestrian_velocity,
        pedestrian_target=pedestrian_target,
        move_duration=move_duration,
        grid_scale=grid_scale,
        collision_distance=collision_distance,
        additional_pedestrians=additional,
    )
    start_state: State = (start[0], start[1], 0)
    if start == goal:
        plan = SpaceTimePlan(
            spatial_path=(start,),
            timed_states=(start_state,),
            actions=(),
            planned_wait_actions=0,
            planned_move_actions=0,
            move_duration=move_duration,
            estimated_duration=0.0,
        )
        return SpaceTimePlanningResult(
            plan=plan,
            failure_reason=None,
            first_action_safety=first_action_safety,
            statistics=SpaceTimeSearchStatistics(
                expanded_states=1,
                generated_states=1,
                maximum_time_index_reached=0,
                planning_horizon_reached=max_time_index == 0,
                open_set_exhausted=False,
                goal_reached=True,
                returned_path_length=0,
                planned_wait_count=0,
            ),
        )

    tie_breaker = count()
    frontier: list[tuple[float, int, float, State]] = []
    heappush(
        frontier,
        (float(_manhattan(start, goal)), next(tie_breaker), 0.0, start_state),
    )
    cost_so_far = {start_state: 0.0}
    came_from: dict[State, tuple[State, SpaceTimeAction]] = {}
    expanded_states = 0
    generated_states = 1
    maximum_time_index_reached = 0

    while frontier:
        _, _, current_cost, current = heappop(frontier)
        if current_cost != cost_so_far[current]:
            continue
        expanded_states += 1
        maximum_time_index_reached = max(
            maximum_time_index_reached,
            current[2],
        )

        current_coordinate = (current[0], current[1])
        if current_coordinate == goal:
            plan = _reconstruct_plan(
                came_from,
                start_state,
                current,
                move_duration,
            )
            return SpaceTimePlanningResult(
                plan=plan,
                failure_reason=None,
                first_action_safety=first_action_safety,
                statistics=SpaceTimeSearchStatistics(
                    expanded_states=expanded_states,
                    generated_states=generated_states,
                    maximum_time_index_reached=maximum_time_index_reached,
                    planning_horizon_reached=(
                        maximum_time_index_reached >= max_time_index
                    ),
                    open_set_exhausted=False,
                    goal_reached=True,
                    returned_path_length=len(plan.actions),
                    planned_wait_count=plan.planned_wait_actions,
                ),
            )
        if current[2] >= max_time_index:
            continue

        for action, (delta_x, delta_y) in _ACTIONS:
            neighbor = (
                current[0] + delta_x,
                current[1] + delta_y,
            )
            if not grid_map.is_inside(neighbor) or not grid_map.is_free(neighbor):
                continue
            if not is_space_time_action_safe(
                current_coordinate,
                neighbor,
                current[2],
                pedestrian_position=pedestrian_position,
                pedestrian_velocity=pedestrian_velocity,
                pedestrian_target=pedestrian_target,
                move_duration=move_duration,
                grid_scale=grid_scale,
                collision_distance=collision_distance,
                additional_pedestrians=additional,
            ):
                continue

            next_state: State = (neighbor[0], neighbor[1], current[2] + 1)
            social_penalty = compute_time_aligned_social_cost(
                neighbor,
                next_state[2],
                pedestrian_position=pedestrian_position,
                pedestrian_velocity=pedestrian_velocity,
                pedestrian_target=pedestrian_target,
                move_duration=move_duration,
                grid_scale=grid_scale,
                social_distance=social_distance,
                social_weight=social_weight,
                additional_pedestrians=additional,
            )
            new_cost = current_cost + 1.0 + social_penalty
            if new_cost >= cost_so_far.get(next_state, float("inf")):
                continue

            cost_so_far[next_state] = new_cost
            came_from[next_state] = (current, action)
            generated_states += 1
            maximum_time_index_reached = max(
                maximum_time_index_reached,
                next_state[2],
            )
            priority = new_cost + _manhattan(neighbor, goal)
            heappush(
                frontier,
                (priority, next(tie_breaker), new_cost, next_state),
            )

    planning_horizon_reached = (
        maximum_time_index_reached >= max_time_index
    )
    pedestrians = _collect_pedestrian_states(
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        additional,
    )
    start_world = grid_to_world(start, grid_scale)
    start_separation = min(
        (
            hypot(
                start_world[0] - position[0],
                start_world[1] - position[1],
            )
            for position, _, _ in pedestrians
        ),
        default=inf,
    )
    has_safe_first_action = any(
        detail.rejection_reason is None for detail in first_action_safety
    )
    if start_separation <= collision_distance:
        failure_reason: SpaceTimeFailureReason = (
            "start_in_predicted_collision"
        )
    elif not has_safe_first_action:
        failure_reason = "no_safe_first_action"
    elif not _is_static_goal_reachable(grid_map, start, goal):
        failure_reason = "goal_unreachable_static"
    elif planning_horizon_reached:
        failure_reason = "time_horizon_exhausted"
    elif expanded_states > 0:
        failure_reason = "search_exhausted"
    else:
        failure_reason = "other"

    return SpaceTimePlanningResult(
        plan=None,
        failure_reason=failure_reason,
        first_action_safety=first_action_safety,
        statistics=SpaceTimeSearchStatistics(
            expanded_states=expanded_states,
            generated_states=generated_states,
            maximum_time_index_reached=maximum_time_index_reached,
            planning_horizon_reached=planning_horizon_reached,
            open_set_exhausted=True,
            goal_reached=False,
            returned_path_length=None,
            planned_wait_count=0,
        ),
    )


def _is_static_goal_reachable(
    grid_map: GridMap,
    start: Coordinate,
    goal: Coordinate,
) -> bool:
    frontier = [start]
    visited = {start}
    while frontier:
        current = frontier.pop()
        if current == goal:
            return True
        for _, (delta_x, delta_y) in _ACTIONS[:-1]:
            neighbor = (current[0] + delta_x, current[1] + delta_y)
            if (
                neighbor not in visited
                and grid_map.is_inside(neighbor)
                and grid_map.is_free(neighbor)
            ):
                visited.add(neighbor)
                frontier.append(neighbor)
    return False


def _validate_endpoint(
    grid_map: GridMap,
    coordinate: Coordinate,
    name: str,
) -> None:
    if not grid_map.is_inside(coordinate):
        raise ValueError(f"{name} must be inside the map")
    if not grid_map.is_free(coordinate):
        raise ValueError(f"{name} must be free")


def _manhattan(first: Coordinate, second: Coordinate) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _reconstruct_plan(
    came_from: dict[State, tuple[State, SpaceTimeAction]],
    start: State,
    goal: State,
    move_duration: float,
) -> SpaceTimePlan:
    states = [goal]
    actions: list[SpaceTimeAction] = []
    current = goal

    while current != start:
        previous, action = came_from[current]
        states.append(previous)
        actions.append(action)
        current = previous

    states.reverse()
    actions.reverse()
    action_tuple = tuple(actions)
    timed_states = tuple(states)
    return SpaceTimePlan(
        spatial_path=tuple((state[0], state[1]) for state in timed_states),
        timed_states=timed_states,
        actions=action_tuple,
        planned_wait_actions=action_tuple.count("WAIT"),
        planned_move_actions=len(action_tuple) - action_tuple.count("WAIT"),
        move_duration=move_duration,
        estimated_duration=len(action_tuple) * move_duration,
    )