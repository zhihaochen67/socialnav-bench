import pytest

from socialnav.env.demo_map import (
    CELL_SIZE,
    GOAL,
    OBSTACLES,
    PEDESTRIAN_PLANNING_CELL,
    SOCIAL_DISTANCE,
    SOCIAL_WEIGHT,
    START,
    build_demo_grid,
    grid_to_world,
    interpolate_path,
    path_minimum_clearance,
)
from socialnav.planners.astar import astar
from socialnav.planners.social_planner import social_astar


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


def test_path_minimum_clearance_uses_world_coordinates() -> None:
    clearance = path_minimum_clearance(
        [(0, 0), (1, 0), (2, 0)],
        pedestrian_position=(1.5, 0.75),
    )

    assert clearance == pytest.approx(CELL_SIZE)


def test_path_minimum_clearance_rejects_empty_path() -> None:
    with pytest.raises(ValueError, match="path must not be empty"):
        path_minimum_clearance([], pedestrian_position=(0.0, 0.0))


def test_demo_social_path_is_longer_and_has_greater_clearance() -> None:
    grid_map = build_demo_grid()
    pedestrian_position = grid_to_world(PEDESTRIAN_PLANNING_CELL)

    astar_path = astar(grid_map, START, GOAL)
    social_path = social_astar(
        grid_map,
        START,
        GOAL,
        pedestrian_positions=[pedestrian_position],
        social_distance=SOCIAL_DISTANCE,
        social_weight=SOCIAL_WEIGHT,
        grid_scale=CELL_SIZE,
    )

    assert astar_path is not None
    assert social_path is not None
    assert len(astar_path) - 1 == 18
    assert len(social_path) - 1 == 20
    assert PEDESTRIAN_PLANNING_CELL in astar_path
    assert PEDESTRIAN_PLANNING_CELL not in social_path

    astar_clearance = path_minimum_clearance(
        astar_path,
        pedestrian_position,
    )
    social_clearance = path_minimum_clearance(
        social_path,
        pedestrian_position,
    )
    assert astar_clearance == 0.0
    assert social_clearance == pytest.approx(CELL_SIZE)
    assert social_clearance > astar_clearance


def test_demo_social_path_is_deterministic() -> None:
    grid_map = build_demo_grid()
    pedestrian_position = grid_to_world(PEDESTRIAN_PLANNING_CELL)

    paths = [
        social_astar(
            grid_map,
            START,
            GOAL,
            pedestrian_positions=[pedestrian_position],
            social_distance=SOCIAL_DISTANCE,
            social_weight=SOCIAL_WEIGHT,
            grid_scale=CELL_SIZE,
        )
        for _ in range(5)
    ]

    assert paths == [paths[0]] * 5
