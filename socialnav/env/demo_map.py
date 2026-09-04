"""Pure grid helpers for the deterministic navigation demo."""

from collections.abc import Sequence
from math import hypot

from socialnav.env.grid_map import Coordinate, GridMap

GRID_WIDTH = 8
GRID_HEIGHT = 6
CELL_SIZE = 0.75
START: Coordinate = (0, 0)
GOAL: Coordinate = (7, 5)
PEDESTRIAN_PLANNING_CELL: Coordinate = (1, 4)
SOCIAL_DISTANCE = 0.70
SOCIAL_WEIGHT = 10.0
OBSTACLES: tuple[Coordinate, ...] = (
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
)

WorldCoordinate = tuple[float, float]


def build_demo_grid() -> GridMap:
    """Return the fixed obstacle map used by the PyBullet demo."""
    grid_map = GridMap(width=GRID_WIDTH, height=GRID_HEIGHT)
    for coordinate in OBSTACLES:
        grid_map.add_obstacle(coordinate)
    return grid_map


def grid_to_world(
    coordinate: Coordinate, cell_size: float = CELL_SIZE
) -> WorldCoordinate:
    """Map a grid cell to the center of its PyBullet ground-plane cell."""
    return coordinate[0] * cell_size, coordinate[1] * cell_size


def path_minimum_clearance(
    path: Sequence[Coordinate],
    pedestrian_position: WorldCoordinate,
    cell_size: float = CELL_SIZE,
) -> float:
    """Return the closest world-space distance from a path to a pedestrian."""
    if not path:
        raise ValueError("path must not be empty")

    world_path = (
        grid_to_world(coordinate, cell_size) for coordinate in path
    )
    return min(
        hypot(x - pedestrian_position[0], y - pedestrian_position[1])
        for x, y in world_path
    )


def interpolate_path(
    path: Sequence[Coordinate],
    steps_per_cell: int,
    cell_size: float = CELL_SIZE,
) -> list[WorldCoordinate]:
    """Convert a grid path to evenly spaced world-coordinate positions."""
    if steps_per_cell <= 0:
        raise ValueError("steps_per_cell must be positive")
    if not path:
        return []

    positions = [grid_to_world(path[0], cell_size)]
    for start, end in zip(path, path[1:]):
        start_x, start_y = grid_to_world(start, cell_size)
        end_x, end_y = grid_to_world(end, cell_size)
        for step in range(1, steps_per_cell + 1):
            fraction = step / steps_per_cell
            positions.append(
                (
                    start_x + (end_x - start_x) * fraction,
                    start_y + (end_y - start_y) * fraction,
                )
            )

    return positions
