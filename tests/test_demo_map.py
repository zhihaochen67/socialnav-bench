import pytest

from socialnav.env.demo_map import (
    CELL_SIZE,
    GOAL,
    OBSTACLES,
    START,
    build_demo_grid,
    grid_to_world,
    interpolate_path,
)
from socialnav.planners.astar import astar


def test_demo_map_has_a_collision_free_path() -> None:
    grid_map = build_demo_grid()

    path = astar(grid_map, START, GOAL)

    assert path is not None
    assert path[0] == START
    assert path[-1] == GOAL
    assert all(grid_map.is_free(coordinate) for coordinate in path)
    assert all(obstacle not in path for obstacle in OBSTACLES)


def test_grid_to_world_uses_explicit_cell_size() -> None:
    assert grid_to_world((3, 2)) == (3 * CELL_SIZE, 2 * CELL_SIZE)
    assert grid_to_world((3, 2), cell_size=1.0) == (3.0, 2.0)


def test_interpolate_path_produces_smooth_positions_and_exact_endpoints() -> None:
    positions = interpolate_path(
        [(0, 0), (1, 0), (1, 1)],
        steps_per_cell=2,
        cell_size=1.0,
    )

    assert positions == [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (1.0, 1.0),
    ]


def test_interpolate_path_rejects_nonpositive_step_count() -> None:
    with pytest.raises(ValueError, match="steps_per_cell must be positive"):
        interpolate_path([(0, 0), (1, 0)], steps_per_cell=0)
