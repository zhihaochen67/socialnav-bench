import pytest

from socialnav.env.grid_map import GridMap


def test_map_dimensions() -> None:
    grid_map = GridMap(width=5, height=4)

    assert grid_map.width == 5
    assert grid_map.height == 4


def test_cells_are_free_by_default() -> None:
    grid_map = GridMap(width=3, height=3)

    assert grid_map.is_free((1, 1))


def test_add_obstacle_marks_cell_as_occupied() -> None:
    grid_map = GridMap(width=3, height=3)

    grid_map.add_obstacle((1, 2))

    assert not grid_map.is_free((1, 2))
    assert grid_map.get_obstacles() == {(1, 2)}


@pytest.mark.parametrize(
    ("coordinate", "expected"),
    [
        ((0, 0), True),
        ((2, 1), True),
        ((-1, 0), False),
        ((0, -1), False),
        ((3, 0), False),
        ((0, 2), False),
    ],
)
def test_boundary_checking(coordinate: tuple[int, int], expected: bool) -> None:
    grid_map = GridMap(width=3, height=2)

    assert grid_map.is_inside(coordinate) is expected


def test_outside_cell_is_not_free() -> None:
    grid_map = GridMap(width=3, height=3)

    assert not grid_map.is_free((3, 1))


def test_adding_obstacle_outside_map_raises_error() -> None:
    grid_map = GridMap(width=3, height=3)

    with pytest.raises(ValueError, match="outside the map"):
        grid_map.add_obstacle((3, 1))
