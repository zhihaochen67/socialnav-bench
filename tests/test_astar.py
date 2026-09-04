import pytest

from socialnav.env.grid_map import GridMap
from socialnav.planners.astar import astar


def test_direct_path_in_empty_grid() -> None:
    grid_map = GridMap(width=4, height=3)

    assert astar(grid_map, (0, 0), (3, 0)) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
    ]


def test_path_around_obstacle() -> None:
    grid_map = GridMap(width=3, height=3)
    grid_map.add_obstacle((1, 0))

    path = astar(grid_map, (0, 0), (2, 0))

    assert path is not None
    assert (1, 0) not in path
    assert len(path) == 5


def test_returned_path_begins_at_start_and_ends_at_goal() -> None:
    grid_map = GridMap(width=4, height=4)
    start = (0, 1)
    goal = (3, 2)

    path = astar(grid_map, start, goal)

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal


def test_consecutive_path_cells_are_free_four_connected_moves() -> None:
    grid_map = GridMap(width=4, height=4)
    grid_map.add_obstacle((1, 0))
    grid_map.add_obstacle((1, 1))

    path = astar(grid_map, (0, 0), (3, 0))

    assert path is not None
    assert all(grid_map.is_free(coordinate) for coordinate in path)
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(path, path[1:])
    )


def test_unreachable_goal_returns_none() -> None:
    grid_map = GridMap(width=3, height=3)
    grid_map.add_obstacle((1, 0))
    grid_map.add_obstacle((0, 1))

    assert astar(grid_map, (0, 0), (2, 2)) is None


def test_start_equals_goal() -> None:
    grid_map = GridMap(width=3, height=3)

    assert astar(grid_map, (1, 1), (1, 1)) == [(1, 1)]


def test_invalid_start_outside_grid() -> None:
    grid_map = GridMap(width=3, height=3)

    with pytest.raises(ValueError, match="start must be inside the map"):
        astar(grid_map, (-1, 0), (2, 2))


def test_invalid_goal_outside_grid() -> None:
    grid_map = GridMap(width=3, height=3)

    with pytest.raises(ValueError, match="goal must be inside the map"):
        astar(grid_map, (0, 0), (3, 2))


def test_blocked_start() -> None:
    grid_map = GridMap(width=3, height=3)
    grid_map.add_obstacle((0, 0))

    with pytest.raises(ValueError, match="start must be free"):
        astar(grid_map, (0, 0), (2, 2))


def test_blocked_goal() -> None:
    grid_map = GridMap(width=3, height=3)
    grid_map.add_obstacle((2, 2))

    with pytest.raises(ValueError, match="goal must be free"):
        astar(grid_map, (0, 0), (2, 2))
