from math import hypot

import pytest

from socialnav.benchmark import (
    REPLAN_STOP_SECONDS,
    REPLAN_STOP_STEPS,
    SustainedStopReplanPolicy,
    interpolate_replanned_route,
    world_to_nearest_free_cell,
)
from socialnav.env.grid_map import GridMap
from socialnav.env.world import SIMULATION_STEP


def test_default_stop_threshold_is_exactly_half_a_second() -> None:
    assert REPLAN_STOP_SECONDS == 0.50
    assert REPLAN_STOP_STEPS == 120
    assert REPLAN_STOP_STEPS * SIMULATION_STEP == REPLAN_STOP_SECONDS


def test_world_mapping_preserves_an_exact_cell_centre() -> None:
    grid_map = GridMap(4, 3)

    assert world_to_nearest_free_cell(
        grid_map, (1.5, 0.75), 0.75
    ) == (2, 1)


def test_world_mapping_selects_the_nearby_cell_centre() -> None:
    grid_map = GridMap(4, 3)

    assert world_to_nearest_free_cell(
        grid_map, (1.49, 1.51), 1.0
    ) == (1, 2)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ((-10.0, 0.2), (0, 0)),
        ((10.0, 10.0), (2, 1)),
    ],
)
def test_world_mapping_clamps_to_free_boundary_cells(
    position: tuple[float, float],
    expected: tuple[int, int],
) -> None:
    grid_map = GridMap(3, 2)

    selected = world_to_nearest_free_cell(grid_map, position, 1.0)

    assert selected == expected
    assert grid_map.is_inside(selected)
    assert grid_map.is_free(selected)


def test_world_mapping_skips_an_obstacle_at_the_nearest_centre() -> None:
    grid_map = GridMap(3, 3)
    grid_map.add_obstacle((1, 1))

    selected = world_to_nearest_free_cell(grid_map, (1.0, 1.0), 1.0)

    assert selected == (0, 1)
    assert grid_map.is_free(selected)


def test_world_mapping_breaks_exact_ties_by_lower_x_then_lower_y() -> None:
    grid_map = GridMap(2, 2)

    assert world_to_nearest_free_cell(
        grid_map, (0.5, 0.5), 1.0
    ) == (0, 0)


def test_world_mapping_rejects_a_grid_without_free_cells() -> None:
    grid_map = GridMap(1, 1)
    grid_map.add_obstacle((0, 0))

    with pytest.raises(ValueError, match="must contain a free cell"):
        world_to_nearest_free_cell(grid_map, (0.0, 0.0), 1.0)


def test_replanned_route_starts_at_actual_position_without_teleporting() -> None:
    current_position = (0.6, 0.1)
    positions = interpolate_replanned_route(
        current_position,
        [(1, 0), (2, 0)],
        1.0,
        steps_per_cell=4,
    )

    assert positions[0] == current_position
    assert positions[-1] == (2.0, 0.0)
    assert (1.0, 0.0) in positions
    assert max(
        hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(positions, positions[1:])
    ) <= 0.25


def test_policy_does_not_trigger_while_moving_normally() -> None:
    policy = SustainedStopReplanPolicy(threshold_steps=2)

    assert not policy.observe(1.0)
    assert not policy.observe(0.5)
    assert not policy.observe(1.0)


def test_policy_waits_for_the_full_sustained_stop_threshold() -> None:
    policy = SustainedStopReplanPolicy(threshold_steps=3)

    assert not policy.observe(0.0)
    assert not policy.observe(0.0)
    assert policy.observe(0.0)


def test_policy_requires_a_fresh_stop_interval_after_each_trigger() -> None:
    policy = SustainedStopReplanPolicy(threshold_steps=2)

    assert not policy.observe(0.0)
    assert policy.observe(0.0)
    assert not policy.observe(0.0)
    assert policy.observe(0.0)

    assert not policy.observe(0.0)
    assert not policy.observe(1.0)
    assert not policy.observe(0.0)
    assert policy.observe(0.0)


def test_policy_rejects_a_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold_steps must be positive"):
        SustainedStopReplanPolicy(threshold_steps=0)
