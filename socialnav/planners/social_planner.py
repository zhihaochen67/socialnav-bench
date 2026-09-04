"""Deterministic social-aware A* planning on a four-connected grid."""

from __future__ import annotations

from collections.abc import Iterable
from heapq import heappop, heappush
from itertools import count

from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.planners.astar import astar
from socialnav.planners.social_cost import compute_social_cost

WorldPosition = tuple[float, float]

_MOVES: tuple[Coordinate, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


def social_astar(
    grid_map: GridMap,
    start: Coordinate,
    goal: Coordinate,
    pedestrian_positions: Iterable[WorldPosition],
    social_distance: float,
    social_weight: float,
    grid_scale: float = 1.0,
) -> list[Coordinate] | None:
    """Return a path minimizing movement plus distance-based social cost.

    Pedestrian positions are expressed in world coordinates. Grid coordinates
    are multiplied by ``grid_scale`` before social costs are evaluated.
    """
    _validate_endpoint(grid_map, start, "start")
    _validate_endpoint(grid_map, goal, "goal")
    if social_distance <= 0.0:
        raise ValueError("social_distance must be positive")
    if social_weight < 0.0:
        raise ValueError("social_weight must be non-negative")
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")

    pedestrians = tuple(pedestrian_positions)
    if not pedestrians or social_weight == 0.0:
        return astar(grid_map, start, goal)
    if start == goal:
        return [start]

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
            if not grid_map.is_inside(neighbor) or not grid_map.is_free(neighbor):
                continue

            world_position = (
                neighbor[0] * grid_scale,
                neighbor[1] * grid_scale,
            )
            social_penalty = sum(
                compute_social_cost(
                    world_position,
                    pedestrian_position,
                    social_distance,
                    social_weight,
                )
                for pedestrian_position in pedestrians
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
    grid_map: GridMap, coordinate: Coordinate, name: str
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
