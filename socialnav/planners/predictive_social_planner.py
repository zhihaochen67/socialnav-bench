"""Deterministic predictive social A* on a four-connected grid."""

from __future__ import annotations

from collections.abc import Iterable
from heapq import heappop, heappush
from itertools import count

from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.planners.astar import astar
from socialnav.planners.pedestrian_prediction import (
    PREDICTION_HORIZONS,
    PedestrianPredictionState,
    compute_predictive_social_cost,
    predict_pedestrian_positions,
)

WorldPosition = tuple[float, float]

_MOVES: tuple[Coordinate, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


def predictive_social_astar(
    grid_map: GridMap,
    start: Coordinate,
    goal: Coordinate,
    pedestrian_position: WorldPosition | None,
    pedestrian_velocity: WorldPosition,
    social_distance: float,
    social_weight: float,
    grid_scale: float = 1.0,
    *,
    pedestrian_target: WorldPosition | None = None,
    horizons: Iterable[float] = PREDICTION_HORIZONS,
    temporal_weights: Iterable[float] | None = None,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> list[Coordinate] | None:
    """Plan using decayed social costs at current and predicted positions.

    Each pedestrian contributes its own independent current-and-predicted
    social cost, and the candidate cost is the unnormalized sum across
    pedestrians.  With no additional pedestrians this is the original
    single-human planner unchanged.
    """
    _validate_endpoint(grid_map, start, "start")
    _validate_endpoint(grid_map, goal, "goal")
    if social_distance <= 0.0:
        raise ValueError("social_distance must be positive")
    if social_weight < 0.0:
        raise ValueError("social_weight must be non-negative")
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")

    pedestrian_states: list[PedestrianPredictionState] = []
    if pedestrian_position is not None:
        pedestrian_states.append(
            (pedestrian_position, pedestrian_velocity, pedestrian_target)
        )
    pedestrian_states.extend(additional_pedestrians)

    if not pedestrian_states or social_weight == 0.0:
        return astar(grid_map, start, goal)
    if start == goal:
        return [start]

    predicted_positions_by_pedestrian = [
        predict_pedestrian_positions(
            position,
            velocity,
            horizons,
            target=target,
        )
        for position, velocity, target in pedestrian_states
    ]
    resolved_temporal_weights = (
        None if temporal_weights is None else tuple(temporal_weights)
    )

    tie_breaker = count()
    frontier: list[tuple[float, int, float, Coordinate]] = []
    heappush(
        frontier,
        (float(_manhattan(start, goal)), next(tie_breaker), 0.0, start),
    )

    came_from: dict[Coordinate, Coordinate] = {}
    cost_so_far = {start: 0.0}

    while frontier:
        _, _, current_cost, current = heappop(frontier)
        if current_cost != cost_so_far[current]:
            continue
        if current == goal:
            return _reconstruct_path(came_from, start, goal)

        for dx, dy in _MOVES:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                not grid_map.is_inside(neighbor)
                or not grid_map.is_free(neighbor)
            ):
                continue

            world_position = (
                neighbor[0] * grid_scale,
                neighbor[1] * grid_scale,
            )
            social_penalty = sum(
                compute_predictive_social_cost(
                    world_position,
                    position,
                    predicted_positions,
                    social_distance,
                    social_weight,
                    resolved_temporal_weights,
                )
                for (position, _, _), predicted_positions in zip(
                    pedestrian_states,
                    predicted_positions_by_pedestrian,
                )
            )
            new_cost = current_cost + 1.0 + social_penalty
            if new_cost >= cost_so_far.get(neighbor, float("inf")):
                continue

            cost_so_far[neighbor] = new_cost
            came_from[neighbor] = current
            priority = new_cost + _manhattan(neighbor, goal)
            heappush(
                frontier,
                (priority, next(tie_breaker), new_cost, neighbor),
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


def _reconstruct_path(
    came_from: dict[Coordinate, Coordinate],
    start: Coordinate,
    goal: Coordinate,
) -> list[Coordinate]:
    path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path