from math import hypot

from socialnav.env.grid_map import GridMap
from socialnav.planners.astar import astar
from socialnav.planners.predictive_social_planner import (
    predictive_social_astar,
)
from socialnav.planners.social_planner import social_astar


def _assert_valid_path(
    grid_map: GridMap,
    path: list[tuple[int, int]] | None,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> None:
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert all(grid_map.is_free(cell) for cell in path)
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(path, path[1:])
    )


def test_empty_future_prediction_matches_current_social_astar() -> None:
    grid_map = GridMap(5, 5)
    arguments = (
        grid_map,
        (0, 2),
        (4, 2),
        (2.0, 2.0),
        (1.0, 0.0),
        2.0,
        10.0,
    )

    predictive_path = predictive_social_astar(*arguments, horizons=())
    social_path = social_astar(
        grid_map,
        (0, 2),
        (4, 2),
        [(2.0, 2.0)],
        2.0,
        10.0,
    )

    assert predictive_path == social_path


def test_zero_social_weight_matches_astar() -> None:
    grid_map = GridMap(5, 4)
    grid_map.add_obstacle((1, 1))

    path = predictive_social_astar(
        grid_map,
        (0, 2),
        (4, 2),
        (2.0, 3.0),
        (0.0, -1.0),
        2.0,
        0.0,
        pedestrian_target=(2.0, 0.0),
    )

    assert path == astar(grid_map, (0, 2), (4, 2))


def test_predictive_path_is_four_connected_and_respects_obstacles() -> None:
    grid_map = GridMap(5, 5)
    grid_map.add_obstacle((1, 2))
    grid_map.add_obstacle((2, 2))

    path = predictive_social_astar(
        grid_map,
        (0, 2),
        (4, 2),
        (3.0, 4.0),
        (0.0, -1.0),
        1.5,
        5.0,
        pedestrian_target=(3.0, 0.0),
    )

    _assert_valid_path(grid_map, path, (0, 2), (4, 2))
    assert (1, 2) not in path
    assert (2, 2) not in path


def test_future_intersection_is_avoided() -> None:
    grid_map = GridMap(5, 5)
    pedestrian_position = (2.0, 4.0)
    start = (0, 2)
    goal = (4, 2)

    ordinary_social_path = social_astar(
        grid_map,
        start,
        goal,
        [pedestrian_position],
        1.5,
        10.0,
    )
    predictive_path = predictive_social_astar(
        grid_map,
        start,
        goal,
        pedestrian_position,
        (0.0, -2.0),
        1.5,
        10.0,
        pedestrian_target=(2.0, 2.0),
    )

    assert ordinary_social_path is not None
    assert predictive_path is not None
    assert (2, 2) in ordinary_social_path
    assert (2, 2) not in predictive_path
    assert len(predictive_path) > len(ordinary_social_path)
    assert min(
        hypot(cell[0] - 2.0, cell[1] - 2.0)
        for cell in predictive_path
    ) > 0.0


def test_predictive_planner_is_deterministic() -> None:
    grid_map = GridMap(6, 5)
    arguments = (
        grid_map,
        (0, 2),
        (5, 2),
        (3.0, 4.0),
        (0.0, -1.25),
        1.5,
        7.0,
    )

    paths = [
        predictive_social_astar(
            *arguments,
            pedestrian_target=(3.0, 1.0),
        )
        for _ in range(10)
    ]

    assert paths == [paths[0]] * 10

def test_temporal_weight_iterable_is_materialized_once() -> None:
    grid_map = GridMap(5, 5)
    arguments = (
        grid_map,
        (0, 2),
        (4, 2),
        (2.0, 4.0),
        (0.0, -2.0),
        1.5,
        10.0,
    )
    weights = (1.0, 0.75, 0.50, 0.25)

    tuple_path = predictive_social_astar(
        *arguments,
        pedestrian_target=(2.0, 2.0),
        temporal_weights=weights,
    )
    generator_path = predictive_social_astar(
        *arguments,
        pedestrian_target=(2.0, 2.0),
        temporal_weights=(weight for weight in weights),
    )

    assert generator_path == tuple_path
