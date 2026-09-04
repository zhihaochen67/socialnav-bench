"""A* path planning for a four-connected grid."""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count

from socialnav.env.grid_map import GridMap

Coordinate = tuple[int, int]

_MOVES: tuple[Coordinate, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


def astar(
    grid_map: GridMap, start: Coordinate, goal: Coordinate
) -> list[Coordinate] | None:
    """Return a shortest four-connected path from start to goal.

    The returned path includes both endpoints. None is returned when the goal
    is unreachable. Invalid or blocked endpoints raise ValueError.
    """
    _validate_endpoint(grid_map, start, "start")
    _validate_endpoint(grid_map, goal, "goal")

    if start == goal:
        return [start]

    tie_breaker = count()
    frontier: list[tuple[int, int, int, Coordinate]] = []
    heappush(frontier, (_manhattan(start, goal), next(tie_breaker), 0, start))

    came_from: dict[Coordinate, Coordinate] = {}
    cost_so_far = {start: 0}

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

            new_cost = current_cost + 1
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
