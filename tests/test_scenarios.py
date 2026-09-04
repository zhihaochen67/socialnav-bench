import pytest

import socialnav.benchmark.scenario as scenario_module
from socialnav.benchmark.scenario import (
    DIVERSE_MIN_PATH_MOVES,
    build_scenario_grid,
    generate_diverse_scenarios,
    generate_scenarios,
    is_pedestrian_route_relevant,
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


def test_controlled_scenario_generation_remains_reproducible() -> None:
    assert generate_scenarios(20, 42) == generate_scenarios(20, 42)


def test_diverse_generator_returns_requested_count() -> None:
    assert len(generate_diverse_scenarios(25, seed=42)) == 25


def test_diverse_generator_is_reproducible_for_same_seed() -> None:
    first = generate_diverse_scenarios(25, seed=42)
    second = generate_diverse_scenarios(25, seed=42)

    assert second == first


def test_different_seeds_change_diverse_scenarios() -> None:
    first = generate_diverse_scenarios(25, seed=1)
    second = generate_diverse_scenarios(25, seed=2)

    assert second != first


def test_diverse_start_and_goal_are_valid_distinct_and_reachable() -> None:
    for scenario in generate_diverse_scenarios(100, seed=42):
        grid_map = build_scenario_grid(scenario)
        path = astar(grid_map, scenario.start, scenario.goal)

        assert scenario.start != scenario.goal
        assert grid_map.is_free(scenario.start)
        assert grid_map.is_free(scenario.goal)
        assert path is not None


def test_diverse_paths_meet_minimum_length_rule() -> None:
    for scenario in generate_diverse_scenarios(100, seed=42):
        path = astar(
            build_scenario_grid(scenario),
            scenario.start,
            scenario.goal,
        )

        assert path is not None
        assert len(path) - 1 >= DIVERSE_MIN_PATH_MOVES


def test_diverse_pedestrian_motion_is_valid_and_non_degenerate() -> None:
    for scenario in generate_diverse_scenarios(100, seed=42):
        grid_map = build_scenario_grid(scenario)
        pedestrian_start = (
            round(scenario.pedestrian_start[0] / scenario.grid_scale),
            round(scenario.pedestrian_start[1] / scenario.grid_scale),
        )
        pedestrian_target = (
            round(scenario.pedestrian_target[0] / scenario.grid_scale),
            round(scenario.pedestrian_target[1] / scenario.grid_scale),
        )

        assert grid_map.is_free(pedestrian_start)
        assert grid_map.is_free(pedestrian_target)
        assert pedestrian_start != pedestrian_target
        assert scenario.pedestrian_speed > 0.0


def test_diverse_pedestrian_motion_is_relevant_to_astar_route() -> None:
    for scenario in generate_diverse_scenarios(100, seed=42):
        path = astar(
            build_scenario_grid(scenario),
            scenario.start,
            scenario.goal,
        )

        assert path is not None
        assert is_pedestrian_route_relevant(scenario, path)


def test_diverse_generation_does_not_consult_social_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("social planner must not select diverse scenarios")

    monkeypatch.setattr(scenario_module, "social_astar", fail_if_called)

    assert len(generate_diverse_scenarios(10, seed=42)) == 10


def test_representative_diverse_batch_has_multiple_distinct_values() -> None:
    scenarios = generate_diverse_scenarios(100, seed=42)

    assert len({scenario.start for scenario in scenarios}) > 1
    assert len({scenario.goal for scenario in scenarios}) > 1
    assert len({scenario.pedestrian_start for scenario in scenarios}) > 1
    assert len({scenario.pedestrian_target for scenario in scenarios}) > 1
    assert len({scenario.pedestrian_speed for scenario in scenarios}) > 1


def test_negative_diverse_scenario_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="count must be non-negative"):
        generate_diverse_scenarios(-1, seed=42)
