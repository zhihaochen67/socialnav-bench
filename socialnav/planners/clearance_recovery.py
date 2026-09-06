"""Deterministic local recovery paths that increase pedestrian clearance."""

from __future__ import annotations

from collections import deque
from math import hypot

from socialnav.env.demo_map import grid_to_world
from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.metrics import Position

from .directional_avoidance import is_separation_increasing

CLEARANCE_EPSILON = 1e-12


def is_clearance_safe_motion(
    robot_position: Position,
    pedestrian_position: Position,
    intended_motion: Position,
    *,
    tolerance: float = CLEARANCE_EPSILON,
) -> bool:
    """Return whether motion does not point meaningfully toward the pedestrian."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    to_human = (
        pedestrian_position[0] - robot_position[0],
        pedestrian_position[1] - robot_position[1],
    )
    if hypot(*to_human) == 0.0 or hypot(*intended_motion) == 0.0:
        return False

    direction_dot = (
        intended_motion[0] * to_human[0]
        + intended_motion[1] * to_human[1]
    )
    return direction_dot <= tolerance


_MOVES: tuple[Coordinate, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


def _clearance(
    coordinate: Coordinate,
    pedestrian_position: Position,
    grid_scale: float,
) -> float:
    world_position = grid_to_world(coordinate, grid_scale)
    return hypot(
        world_position[0] - pedestrian_position[0],
        world_position[1] - pedestrian_position[1],
    )


def _neighbors(grid_map: GridMap, coordinate: Coordinate) -> list[Coordinate]:
    neighbors = []
    for delta_x, delta_y in _MOVES:
        candidate = (
            coordinate[0] + delta_x,
            coordinate[1] + delta_y,
        )
        if grid_map.is_free(candidate):
            neighbors.append(candidate)
    return neighbors


def _ordered_transitions(
    grid_map: GridMap,
    coordinate: Coordinate,
    pedestrian_position: Position,
    grid_scale: float,
    current_clearance: float,
) -> list[Coordinate]:
    candidates = [
        candidate
        for candidate in _neighbors(grid_map, coordinate)
        if (
            _clearance(candidate, pedestrian_position, grid_scale)
            >= current_clearance - CLEARANCE_EPSILON
        )
    ]
    return sorted(
        candidates,
        key=lambda candidate: (
            not (
                _clearance(candidate, pedestrian_position, grid_scale)
                > current_clearance + CLEARANCE_EPSILON
            ),
            -_clearance(candidate, pedestrian_position, grid_scale),
            candidate[0],
            candidate[1],
        ),
    )


def _first_target_is_safe(
    target: Coordinate,
    robot_position: Position,
    pedestrian_position: Position,
    grid_scale: float,
    robot_clearance: float,
) -> bool:
    target_position = grid_to_world(target, grid_scale)
    intended_motion = (
        target_position[0] - robot_position[0],
        target_position[1] - robot_position[1],
    )
    return (
        is_separation_increasing(
            robot_position,
            pedestrian_position,
            intended_motion,
            tolerance=CLEARANCE_EPSILON,
        )
        and _clearance(target, pedestrian_position, grid_scale)
        >= robot_clearance - CLEARANCE_EPSILON
    )


def find_clearance_recovery_path(
    grid_map: GridMap,
    start: Coordinate,
    robot_position: Position,
    pedestrian_position: Position,
    grid_scale: float,
    target_clearance: float,
) -> list[Coordinate] | None:
    """Find a short deterministic route to greater pedestrian clearance.

    Returned coordinates are physical targets and omit the mapped start unless
    moving to its centre is itself a safe, nonzero first motion. Every later
    transition is four-connected and may reduce clearance by at most the fixed
    clearance epsilon. The first target always passes the same strict
    away-direction test as the directional controller.

    Breadth-first search stops at the first path depth that reaches the target
    clearance. At that depth, greater clearance wins, followed by lower x and
    lower y. If the target is unreachable, the fallback is maximum reachable
    clearance, then shortest path, lower coordinates, and finally
    lexicographic path order.
    """
    if not grid_map.is_inside(start):
        raise ValueError("start must be inside the map")
    if not grid_map.is_free(start):
        raise ValueError("start must be free")
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")
    if target_clearance <= 0.0:
        raise ValueError("target_clearance must be positive")

    robot_clearance = hypot(
        robot_position[0] - pedestrian_position[0],
        robot_position[1] - pedestrian_position[1],
    )
    if robot_clearance == 0.0:
        return None
    if robot_clearance >= target_clearance - CLEARANCE_EPSILON:
        return []

    possible_first_targets = [start, *_neighbors(grid_map, start)]
    first_targets = sorted(
        {
            target
            for target in possible_first_targets
            if _first_target_is_safe(
                target,
                robot_position,
                pedestrian_position,
                grid_scale,
                robot_clearance,
            )
        },
        key=lambda target: (
            not (
                _clearance(target, pedestrian_position, grid_scale)
                > robot_clearance + CLEARANCE_EPSILON
            ),
            -_clearance(target, pedestrian_position, grid_scale),
            target[0],
            target[1],
        ),
    )
    if not first_targets:
        return None

    queue: deque[tuple[Coordinate, tuple[Coordinate, ...]]] = deque()
    visited: set[Coordinate] = set()
    reachable: list[tuple[Coordinate, tuple[Coordinate, ...], float]] = []
    for target in first_targets:
        visited.add(target)
        path = (target,)
        queue.append((target, path))
        reachable.append(
            (
                target,
                path,
                _clearance(target, pedestrian_position, grid_scale),
            )
        )

    while queue:
        depth = len(queue[0][1])
        layer: list[tuple[Coordinate, tuple[Coordinate, ...], float]] = []
        while queue and len(queue[0][1]) == depth:
            coordinate, path = queue.popleft()
            layer.append(
                (
                    coordinate,
                    path,
                    _clearance(
                        coordinate,
                        pedestrian_position,
                        grid_scale,
                    ),
                )
            )

        targets = [
            candidate
            for candidate in layer
            if candidate[2] >= target_clearance - CLEARANCE_EPSILON
        ]
        if targets:
            _, path, _ = min(
                targets,
                key=lambda candidate: (
                    -candidate[2],
                    candidate[0][0],
                    candidate[0][1],
                    candidate[1],
                ),
            )
            return list(path)

        for coordinate, path, current_clearance in layer:
            for neighbor in _ordered_transitions(
                grid_map,
                coordinate,
                pedestrian_position,
                grid_scale,
                current_clearance,
            ):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                neighbor_path = (*path, neighbor)
                neighbor_clearance = _clearance(
                    neighbor,
                    pedestrian_position,
                    grid_scale,
                )
                queue.append((neighbor, neighbor_path))
                reachable.append(
                    (neighbor, neighbor_path, neighbor_clearance)
                )

    _, fallback_path, _ = min(
        reachable,
        key=lambda candidate: (
            -candidate[2],
            len(candidate[1]),
            candidate[0][0],
            candidate[0][1],
            candidate[1],
        ),
    )
    return list(fallback_path)

def is_multi_clearance_safe_motion(
    robot_position: Position,
    pedestrian_positions: Iterable[Position],
    intended_motion: Position,
    *,
    tolerance: float = CLEARANCE_EPSILON,
) -> bool:
    """Return whether motion is clearance-safe toward every pedestrian.

    Every pedestrian must individually pass the existing single-human
    clearance check, so an escape from one pedestrian can never move
    dangerously toward another.  With one pedestrian this reproduces
    :func:`is_clearance_safe_motion` exactly, and with none it is
    unconstrained.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    return all(
        is_clearance_safe_motion(
            robot_position,
            pedestrian_position,
            intended_motion,
            tolerance=tolerance,
        )
        for pedestrian_position in pedestrian_positions
    )