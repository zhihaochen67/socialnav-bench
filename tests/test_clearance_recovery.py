from math import hypot

import pytest

from socialnav.env.demo_map import grid_to_world
from socialnav.env.grid_map import GridMap
from socialnav.planners.clearance_recovery import (
    CLEARANCE_EPSILON,
    find_clearance_recovery_path,
    is_clearance_safe_motion,
)


def _human_clearance(
    coordinate: tuple[int, int],
    human: tuple[float, float],
    grid_scale: float = 1.0,
) -> float:
    position = grid_to_world(coordinate, grid_scale)
    return hypot(position[0] - human[0], position[1] - human[1])


def _assert_grid_path_is_valid(
    grid_map: GridMap,
    start: tuple[int, int],
    path: list[tuple[int, int]],
) -> None:
    assert path
    assert all(grid_map.is_inside(cell) for cell in path)
    assert all(grid_map.is_free(cell) for cell in path)
    route = path if path[0] == start else [start, *path]
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(route, route[1:])
    )


def test_simple_free_map_finds_an_away_path_to_target_clearance() -> None:
    grid_map = GridMap(5, 3)

    path = find_clearance_recovery_path(
        grid_map,
        (2, 1),
        (2.0, 1.0),
        (2.4, 1.0),
        1.0,
        1.25,
    )

    assert path == [(1, 1)]
    _assert_grid_path_is_valid(grid_map, (2, 1), path)
    assert _human_clearance(path[-1], (2.4, 1.0)) >= 1.25


def test_recovery_path_is_four_connected_free_and_monotonic() -> None:
    grid_map = GridMap(6, 5)
    robot = (2.0, 2.0)
    human = (4.0, 2.0)

    path = find_clearance_recovery_path(
        grid_map,
        (2, 2),
        robot,
        human,
        1.0,
        3.5,
    )

    assert path is not None
    _assert_grid_path_is_valid(grid_map, (2, 2), path)
    clearances = [
        hypot(robot[0] - human[0], robot[1] - human[1]),
        *(_human_clearance(cell, human) for cell in path),
    ]
    assert all(
        next_clearance
        >= current_clearance - CLEARANCE_EPSILON
        for current_clearance, next_clearance in zip(
            clearances,
            clearances[1:],
        )
    )
    assert clearances[-1] >= 3.5


def test_obstacle_blocks_direct_escape_but_alternate_route_exists() -> None:
    grid_map = GridMap(5, 5)
    grid_map.add_obstacle((1, 2))
    human = (3.0, 3.0)

    path = find_clearance_recovery_path(
        grid_map,
        (2, 2),
        (2.0, 2.0),
        human,
        1.0,
        3.5,
    )

    assert path is not None
    assert path[0] == (2, 1)
    assert (1, 2) not in path
    _assert_grid_path_is_valid(grid_map, (2, 2), path)
    assert _human_clearance(path[-1], human) >= 3.5


def test_no_safe_initial_recovery_route_returns_none() -> None:
    grid_map = GridMap(3, 3)
    grid_map.add_obstacle((0, 1))

    assert (
        find_clearance_recovery_path(
            grid_map,
            (1, 1),
            (1.0, 1.0),
            (2.0, 1.0),
            1.0,
            2.0,
        )
        is None
    )


def test_first_physical_motion_points_away_from_human() -> None:
    grid_map = GridMap(4, 3)
    robot = (1.2, 1.0)
    human = (1.5, 1.0)

    path = find_clearance_recovery_path(
        grid_map,
        (1, 1),
        robot,
        human,
        1.0,
        1.25,
    )

    assert path is not None
    first_target = grid_to_world(path[0], 1.0)
    motion = (
        first_target[0] - robot[0],
        first_target[1] - robot[1],
    )
    to_human = (
        human[0] - robot[0],
        human[1] - robot[1],
    )
    assert motion[0] * to_human[0] + motion[1] * to_human[1] < (
        -CLEARANCE_EPSILON
    )


def test_between_cell_centres_is_handled_without_start_cell_teleport() -> None:
    grid_map = GridMap(4, 3)

    path = find_clearance_recovery_path(
        grid_map,
        (1, 1),
        (1.2, 1.0),
        (1.5, 1.0),
        1.0,
        1.25,
    )

    assert path == [(0, 1)]


def test_toward_only_first_candidate_is_rejected() -> None:
    grid_map = GridMap(2, 1)

    assert (
        find_clearance_recovery_path(
            grid_map,
            (0, 0),
            (0.0, 0.0),
            (1.0, 0.0),
            1.0,
            2.0,
        )
        is None
    )


def test_tangential_only_first_candidate_is_rejected() -> None:
    grid_map = GridMap(1, 2)

    assert (
        find_clearance_recovery_path(
            grid_map,
            (0, 0),
            (0.0, 0.0),
            (1.0, 0.0),
            1.0,
            2.0,
        )
        is None
    )


def test_same_robot_and_human_position_returns_none() -> None:
    grid_map = GridMap(3, 3)

    assert (
        find_clearance_recovery_path(
            grid_map,
            (1, 1),
            (1.0, 1.0),
            (1.0, 1.0),
            1.0,
            2.0,
        )
        is None
    )


def test_ties_choose_lower_x_then_lower_y() -> None:
    grid_map = GridMap(3, 3)

    path = find_clearance_recovery_path(
        grid_map,
        (1, 1),
        (1.0, 1.0),
        (2.0, 2.0),
        1.0,
        2.2,
    )

    assert path == [(0, 1)]


def test_unreachable_target_falls_back_to_maximum_clearance() -> None:
    grid_map = GridMap(3, 1)

    path = find_clearance_recovery_path(
        grid_map,
        (1, 0),
        (1.0, 0.0),
        (1.2, 0.0),
        1.0,
        10.0,
    )

    assert path == [(0, 0)]


def test_recovery_search_is_deterministic() -> None:
    grid_map = GridMap(5, 5)
    arguments = (
        grid_map,
        (2, 2),
        (2.0, 2.0),
        (3.0, 3.0),
        1.0,
        3.5,
    )

    first = find_clearance_recovery_path(*arguments)
    repeated = tuple(
        find_clearance_recovery_path(*arguments) for _ in range(10)
    )

    assert repeated == (first,) * 10


@pytest.mark.parametrize(
    ("grid_scale", "target_clearance", "message"),
    (
        (0.0, 1.25, "grid_scale must be positive"),
        (1.0, 0.0, "target_clearance must be positive"),
    ),
)
def test_recovery_search_validates_positive_distances(
    grid_scale: float,
    target_clearance: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        find_clearance_recovery_path(
            GridMap(2, 2),
            (0, 0),
            (0.0, 0.0),
            (1.0, 0.0),
            grid_scale,
            target_clearance,
        )

@pytest.mark.parametrize(
    ("motion", "expected"),
    (
        ((-1.0, 0.0), True),
        ((0.0, 1.0), True),
        ((1.0, 0.0), False),
        ((0.0, 0.0), False),
    ),
)
def test_recovery_motion_safety_uses_current_geometry(
    motion: tuple[float, float],
    expected: bool,
) -> None:
    assert (
        is_clearance_safe_motion(
            (0.0, 0.0),
            (0.5, 0.0),
            motion,
        )
        is expected
    )


def test_recovery_motion_safety_rejects_coincident_positions() -> None:
    assert not is_clearance_safe_motion(
        (0.0, 0.0),
        (0.0, 0.0),
        (-1.0, 0.0),
    )

