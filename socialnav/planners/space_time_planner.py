"""Deterministic time-expanded social A* with explicit wait actions."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import floor, hypot
from typing import Literal

from socialnav.env.demo_map import grid_to_world
from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.planners.pedestrian_prediction import (
    predict_pedestrian_position_at_time,
)
from socialnav.planners.social_cost import compute_social_cost

State = tuple[int, int, int]
SpaceTimeAction = Literal["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]
Position = tuple[float, float]

_ACTIONS: tuple[tuple[SpaceTimeAction, Coordinate], ...] = (
    ("UP", (0, -1)),
    ("RIGHT", (1, 0)),
    ("DOWN", (0, 1)),
    ("LEFT", (-1, 0)),
    ("WAIT", (0, 0)),
)


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
) -> float:
    """Evaluate social cost at the state's exact estimated arrival time."""
    if time_index < 0:
        raise ValueError("time_index must be non-negative")
    arrival_time = time_index * move_duration
    pedestrian_at_arrival = predict_pedestrian_position_at_time(
        pedestrian_position,
        pedestrian_velocity,
        arrival_time,
        target=pedestrian_target,
    )
    return compute_social_cost(
        grid_to_world(coordinate, grid_scale),
        pedestrian_at_arrival,
        social_distance,
        social_weight,
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
) -> bool:
    """Check start, midpoint, and end separation for one timed action."""
    if start_time_index < 0:
        raise ValueError("start_time_index must be non-negative")
    if move_duration <= 0.0:
        raise ValueError("move_duration must be positive")
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")
    if collision_distance <= 0.0:
        raise ValueError("collision_distance must be positive")

    start_world = grid_to_world(start, grid_scale)
    end_world = grid_to_world(end, grid_scale)
    start_time = start_time_index * move_duration

    for fraction in (0.0, 0.5, 1.0):
        robot_position = (
            start_world[0] + (end_world[0] - start_world[0]) * fraction,
            start_world[1] + (end_world[1] - start_world[1]) * fraction,
        )
        pedestrian_at_time = predict_pedestrian_position_at_time(
            pedestrian_position,
            pedestrian_velocity,
            start_time + move_duration * fraction,
            target=pedestrian_target,
        )
        if hypot(
            robot_position[0] - pedestrian_at_time[0],
            robot_position[1] - pedestrian_at_time[1],
        ) <= collision_distance:
            return False

    return True


def space_time_social_astar(
    grid_map: GridMap,
    start: Coordinate,
    goal: Coordinate,
    pedestrian_position: Position,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    social_distance: float,
    social_weight: float,
    collision_distance: float,
    grid_scale: float,
    robot_speed: float,
    max_time_seconds: float,
) -> SpaceTimePlan | None:
    """Plan in (x, y, time_index) using time-aligned pedestrian occupancy."""
    _validate_endpoint(grid_map, start, "start")
    _validate_endpoint(grid_map, goal, "goal")
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

    move_duration = grid_scale / robot_speed
    max_time_index = floor(max_time_seconds / move_duration)
    start_state: State = (start[0], start[1], 0)
    if start == goal:
        return SpaceTimePlan(
            spatial_path=(start,),
            timed_states=(start_state,),
            actions=(),
            planned_wait_actions=0,
            planned_move_actions=0,
            move_duration=move_duration,
            estimated_duration=0.0,
        )

    tie_breaker = count()
    frontier: list[tuple[float, int, float, State]] = []
    heappush(
        frontier,
        (float(_manhattan(start, goal)), next(tie_breaker), 0.0, start_state),
    )
    cost_so_far = {start_state: 0.0}
    came_from: dict[State, tuple[State, SpaceTimeAction]] = {}

    while frontier:
        _, _, current_cost, current = heappop(frontier)
        if current_cost != cost_so_far[current]:
            continue

        current_coordinate = (current[0], current[1])
        if current_coordinate == goal:
            return _reconstruct_plan(
                came_from,
                start_state,
                current,
                move_duration,
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
            )
            new_cost = current_cost + 1.0 + social_penalty
            if new_cost >= cost_so_far.get(next_state, float("inf")):
                continue

            cost_so_far[next_state] = new_cost
            came_from[next_state] = (current, action)
            priority = new_cost + _manhattan(neighbor, goal)
            heappush(
                frontier,
                (priority, next(tie_breaker), new_cost, next_state),
            )

    return None


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
