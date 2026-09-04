import pytest

from socialnav.benchmark.scenario import (
    build_scenario_grid,
    generate_scenarios,
)
from socialnav.planners.astar import astar


def test_generate_scenarios_returns_requested_count() -> None:
    assert len(generate_scenarios(7, seed=42)) == 7


def test_same_seed_reproduces_identical_scenarios() -> None:
    assert generate_scenarios(10, seed=42) == generate_scenarios(10, seed=42)


def test_different_seeds_change_generated_scenarios() -> None:
    first = generate_scenarios(10, seed=1)
    second = generate_scenarios(10, seed=2)

    first_motion = [
        (scenario.pedestrian_start, scenario.pedestrian_target, scenario.pedestrian_speed)
        for scenario in first
    ]
    second_motion = [
        (scenario.pedestrian_start, scenario.pedestrian_target, scenario.pedestrian_speed)
        for scenario in second
    ]
    assert first_motion != second_motion


def test_generated_start_and_goal_are_distinct_valid_free_cells() -> None:
    for scenario in generate_scenarios(10, seed=42):
        grid_map = build_scenario_grid(scenario)

        assert scenario.start != scenario.goal
        assert grid_map.is_free(scenario.start)
        assert grid_map.is_free(scenario.goal)


def test_every_generated_scenario_has_reachable_astar_path() -> None:
    for scenario in generate_scenarios(10, seed=42):
        grid_map = build_scenario_grid(scenario)

        assert astar(grid_map, scenario.start, scenario.goal) is not None


def test_pedestrian_starts_on_astar_path_and_moves_off_route() -> None:
    for scenario in generate_scenarios(10, seed=42):
        grid_map = build_scenario_grid(scenario)
        path = astar(grid_map, scenario.start, scenario.goal)
        assert path is not None

        pedestrian_cell = (
            round(scenario.pedestrian_start[0] / scenario.grid_scale),
            round(scenario.pedestrian_start[1] / scenario.grid_scale),
        )
        target_cell = (
            round(scenario.pedestrian_target[0] / scenario.grid_scale),
            round(scenario.pedestrian_target[1] / scenario.grid_scale),
        )
        direction = (
            (target_cell[0] - pedestrian_cell[0]) // 2,
            (target_cell[1] - pedestrian_cell[1]) // 2,
        )
        first_off_route_cell = (
            pedestrian_cell[0] + direction[0],
            pedestrian_cell[1] + direction[1],
        )
        assert pedestrian_cell in path
        assert target_cell not in path
        assert first_off_route_cell not in path
        assert grid_map.is_free(first_off_route_cell)


def test_negative_scenario_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="count must be non-negative"):
        generate_scenarios(-1, seed=42)
