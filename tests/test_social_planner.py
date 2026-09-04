from math import hypot

import pytest

from socialnav.env.grid_map import GridMap
from socialnav.planners.astar import astar
from socialnav.planners.social_planner import social_astar


def test_empty_pedestrian_list_matches_shortest_path_astar() -> None:
    grid_map = GridMap(width=4, height=3)

    path = social_astar(grid_map, (0, 1), (3, 1), [], 1.0, 5.0)

    assert path == astar(grid_map, (0, 1), (3, 1))


def test_zero_social_weight_matches_shortest_path_astar() -> None:
    grid_map = GridMap(width=5, height=5)
    start = (0, 2)
    goal = (4, 2)

    path = social_astar(
        grid_map,
        start,
        goal,
        pedestrian_positions=[(2.0, 2.0)],
        social_distance=2.0,
        social_weight=0.0,
    )

    assert path == astar(grid_map, start, goal)


def test_high_social_weight_chooses_longer_path_with_more_clearance() -> None:
    grid_map = GridMap(width=5, height=5)
    start = (0, 2)
    goal = (4, 2)
    pedestrian = (2.0, 2.0)

    shortest_path = astar(grid_map, start, goal)
    social_path = social_astar(
        grid_map,
        start,
        goal,
        pedestrian_positions=[pedestrian],
        social_distance=2.0,
        social_weight=10.0,
    )

    assert shortest_path == [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    assert social_path == [
        (0, 2),
        (0, 1),
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
    ]
    assert social_path[0] == start
    assert social_path[-1] == goal
    assert len(social_path) > len(shortest_path)
    assert all(grid_map.is_free(coordinate) for coordinate in social_path)
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(social_path, social_path[1:])
    )

    shortest_clearance = min(
        hypot(x - pedestrian[0], y - pedestrian[1])
        for x, y in shortest_path
    )
    social_clearance = min(
        hypot(x - pedestrian[0], y - pedestrian[1])
        for x, y in social_path
    )
    assert social_clearance > shortest_clearance


def test_path_uses_only_free_four_connected_cells() -> None:
    grid_map = GridMap(width=4, height=4)
    grid_map.add_obstacle((1, 0))
    grid_map.add_obstacle((1, 1))

    path = social_astar(
        grid_map,
        (0, 0),
        (3, 0),
        pedestrian_positions=[(3.0, 3.0)],
        social_distance=1.0,
        social_weight=2.0,
    )

    assert path is not None
    assert all(grid_map.is_free(coordinate) for coordinate in path)
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(path, path[1:])
    )


def test_grid_scale_is_used_for_world_space_social_distance() -> None:
    grid_map = GridMap(width=3, height=2)

    path = social_astar(
        grid_map,
        (0, 0),
        (2, 0),
        pedestrian_positions=[(2.0, 0.0)],
        social_distance=0.75,
        social_weight=10.0,
        grid_scale=2.0,
    )

    assert path == [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]


def test_unreachable_goal_returns_none() -> None:
    grid_map = GridMap(width=3, height=3)
    grid_map.add_obstacle((1, 0))
    grid_map.add_obstacle((0, 1))

    path = social_astar(
        grid_map,
        (0, 0),
        (2, 2),
        pedestrian_positions=[(2.0, 2.0)],
        social_distance=1.0,
        social_weight=2.0,
    )

    assert path is None


def test_start_equals_goal_returns_single_cell_path() -> None:
    grid_map = GridMap(width=3, height=3)

    path = social_astar(
        grid_map,
        (1, 1),
        (1, 1),
        pedestrian_positions=[(1.0, 1.0)],
        social_distance=1.0,
        social_weight=2.0,
    )

    assert path == [(1, 1)]


def test_invalid_start_outside_grid() -> None:
    grid_map = GridMap(width=3, height=3)

    with pytest.raises(ValueError, match="start must be inside the map"):
        social_astar(grid_map, (-1, 0), (2, 2), [], 1.0, 1.0)


def test_invalid_goal_outside_grid() -> None:
    grid_map = GridMap(width=3, height=3)

    with pytest.raises(ValueError, match="goal must be inside the map"):
        social_astar(grid_map, (0, 0), (3, 2), [], 1.0, 1.0)


def test_blocked_start() -> None:
    grid_map = GridMap(width=3, height=3)
    grid_map.add_obstacle((0, 0))

    with pytest.raises(ValueError, match="start must be free"):
        social_astar(grid_map, (0, 0), (2, 2), [], 1.0, 1.0)


def test_blocked_goal() -> None:
    grid_map = GridMap(width=3, height=3)
    grid_map.add_obstacle((2, 2))

    with pytest.raises(ValueError, match="goal must be free"):
        social_astar(grid_map, (0, 0), (2, 2), [], 1.0, 1.0)


@pytest.mark.parametrize("social_distance", [0.0, -1.0])
def test_invalid_social_distance_is_rejected(
    social_distance: float,
) -> None:
    grid_map = GridMap(width=3, height=3)

    with pytest.raises(ValueError, match="social_distance must be positive"):
        social_astar(grid_map, (0, 0), (2, 2), [], social_distance, 1.0)


def test_negative_social_weight_is_rejected() -> None:
    grid_map = GridMap(width=3, height=3)

    with pytest.raises(ValueError, match="social_weight must be non-negative"):
        social_astar(grid_map, (0, 0), (2, 2), [], 1.0, -0.1)


@pytest.mark.parametrize("grid_scale", [0.0, -1.0])
def test_invalid_grid_scale_is_rejected(grid_scale: float) -> None:
    grid_map = GridMap(width=3, height=3)

    with pytest.raises(ValueError, match="grid_scale must be positive"):
        social_astar(
            grid_map,
            (0, 0),
            (2, 2),
            [],
            1.0,
            1.0,
            grid_scale=grid_scale,
        )


def test_repeated_calls_are_deterministic() -> None:
    grid_map = GridMap(width=5, height=5)
    arguments = (
        grid_map,
        (0, 2),
        (4, 2),
        [(2.0, 2.0)],
        2.0,
        10.0,
    )

    paths = [social_astar(*arguments) for _ in range(10)]

    assert paths == [paths[0]] * 10
