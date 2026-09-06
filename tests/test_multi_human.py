"""Phase 7 multi-pedestrian regression tests."""

from math import hypot

import pytest

import socialnav.benchmark.scenario as scenario_module
from socialnav.benchmark import (
    PedestrianSpec,
    Scenario,
    build_scenario_grid,
    generate_diverse_scenarios,
    generate_scenarios,
    run_episode_with_trace,
)
from socialnav.env.grid_map import GridMap
from socialnav.metrics import (
    colliding_human_indices,
    compute_minimum_human_distance,
    compute_per_human_minimum_distances,
    compute_per_human_social_violation_rates,
    compute_social_violation_rate,
    has_human_collision,
)
from socialnav.planners import (
    compute_directional_speed_scale,
    compute_multi_social_cost,
    compute_multi_speed_scale,
    compute_time_aligned_social_cost,
    is_multi_collision_egress_motion_safe,
    is_space_time_action_safe,
    predict_multi_pedestrian_positions,
)
from socialnav.planners.dynamic_avoidance import compute_speed_scale
from socialnav.planners.robust_space_time_planner import (
    build_continuous_start_transitions,
    select_continuous_start_bridge,
)
from socialnav.planners.space_time_planner import (
    diagnose_space_time_social_astar,
)
from socialnav.planners.pedestrian_prediction import (
    predict_pedestrian_position_at_time,
)


# ---------------------------------------------------------------------------
# Part A: Scenario model
# ---------------------------------------------------------------------------

def _scenario(pedestrians: tuple[PedestrianSpec, ...]) -> Scenario:
    return Scenario(
        scenario_id="multi-test",
        grid_width=3,
        grid_height=3,
        obstacle_cells=(),
        start=(0, 0),
        goal=(2, 2),
        grid_scale=1.0,
        pedestrians=pedestrians,
    )


def _spec(
    start: tuple[float, float],
    target: tuple[float, float],
    speed: float = 0.5,
) -> PedestrianSpec:
    return PedestrianSpec(start=start, target=target, speed=speed)


def test_scenario_supports_zero_one_three_and_ten_pedestrians() -> None:
    for count in (0, 1, 3, 10):
        scenario = _scenario(
            tuple(
                _spec((float(index), 0.0), (float(index), 1.0))
                for index in range(count)
            )
        )
        assert scenario.pedestrian_count == count
        assert len(scenario.pedestrians) == count


def test_scenario_legacy_single_human_constructor_remains_compatible() -> None:
    scenario = Scenario(
        scenario_id="legacy",
        grid_width=3,
        grid_height=2,
        obstacle_cells=(),
        start=(0, 0),
        goal=(2, 0),
        grid_scale=1.0,
        pedestrian_start=(1.0, 1.0),
        pedestrian_target=(1.0, 0.0),
        pedestrian_speed=0.5,
    )

    assert scenario.pedestrians == (_spec((1.0, 1.0), (1.0, 0.0)),)
    assert scenario.pedestrian_start == (1.0, 1.0)
    assert scenario.pedestrian_target == (1.0, 0.0)
    assert scenario.pedestrian_speed == 0.5


def test_scenario_rejects_mixed_legacy_and_tuple_construction() -> None:
    with pytest.raises(ValueError, match="not both"):
        Scenario(
            scenario_id="mixed",
            grid_width=3,
            grid_height=2,
            obstacle_cells=(),
            start=(0, 0),
            goal=(2, 0),
            grid_scale=1.0,
            pedestrians=(_spec((0.0, 1.0), (1.0, 1.0)),),
            pedestrian_start=(1.0, 1.0),
            pedestrian_target=(1.0, 0.0),
            pedestrian_speed=0.5,
        )


def test_pedestrian_specs_are_immutable() -> None:
    spec = _spec((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(AttributeError):
        spec.start = (1.0, 1.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Part C: metrics
# ---------------------------------------------------------------------------

def test_any_human_collision_includes_second_human() -> None:
    robot = [(0.0, 0.0), (0.0, 0.0)]
    humans = [[(0.5, 0.0), (0.5, 0.0)], [(0.1, 0.0), (0.1, 0.0)]]

    assert has_human_collision(robot, humans, 0.34)
    assert colliding_human_indices(robot, humans, 0.34) == (1,)


def test_closest_human_determines_minimum_distance() -> None:
    robot = [(0.0, 0.0), (0.0, 0.0)]
    humans = [[(1.0, 0.0), (1.0, 0.0)], [(0.5, 0.0), (0.5, 0.0)]]

    assert compute_minimum_human_distance(robot, humans) == 0.5
    assert compute_per_human_minimum_distances(robot, humans) == (1.0, 0.5)


def test_social_violation_counts_timestep_once_with_multiple_humans() -> None:
    robot = [(0.0, 0.0), (0.0, 0.0), (5.0, 0.0)]
    humans = [
        [(0.1, 0.0), (0.1, 0.0), (6.0, 0.0)],
        [(0.2, 0.0), (0.2, 0.0), (6.2, 0.0)],
    ]

    rate = compute_social_violation_rate(robot, humans, 0.7)

    assert rate == pytest.approx(2.0 / 3.0)
    assert compute_per_human_social_violation_rates(
        robot, humans, 0.7
    ) == pytest.approx((2.0 / 3.0, 2.0 / 3.0))


# ---------------------------------------------------------------------------
# Part D: social cost aggregation
# ---------------------------------------------------------------------------

def test_multi_social_cost_sums_without_normalization() -> None:
    single = compute_multi_social_cost(
        (0.0, 0.0),
        [(0.2, 0.0)],
        0.7,
        10.0,
    )

    assert compute_multi_social_cost(
        (0.0, 0.0),
        [(0.2, 0.0), (0.2, 0.0), (0.2, 0.0)],
        0.7,
        10.0,
    ) == pytest.approx(3.0 * single)


def test_multi_social_cost_is_zero_with_no_pedestrians() -> None:
    assert compute_multi_social_cost((0.0, 0.0), [], 0.7, 10.0) == 0.0


def test_distant_pedestrian_contributes_zero_cost() -> None:
    near = compute_multi_social_cost((0.0, 0.0), [(0.2, 0.0)], 0.7, 10.0)

    assert compute_multi_social_cost(
        (0.0, 0.0),
        [(0.2, 0.0), (5.0, 5.0)],
        0.7,
        10.0,
    ) == pytest.approx(near)


def test_multi_social_cost_matches_single_human_cost_for_n1() -> None:
    from socialnav.planners.social_cost import compute_social_cost

    assert compute_multi_social_cost(
        (0.1, 0.2),
        [(0.4, 0.5)],
        0.7,
        3.0,
    ) == compute_social_cost((0.1, 0.2), (0.4, 0.5), 0.7, 3.0)


# ---------------------------------------------------------------------------
# Part E: prediction
# ---------------------------------------------------------------------------

def test_multi_prediction_is_independent_and_target_clamped() -> None:
    positions = predict_multi_pedestrian_positions(
        (
            ((0.0, 0.0), (1.0, 0.0), None),
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.5)),
        ),
        10.0,
    )

    assert positions[0] == (10.0, 0.0)
    assert positions[1] == (0.0, 1.5)


def test_multi_prediction_is_deterministic() -> None:
    states = (
        ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0)),
        ((0.0, 2.0), (0.0, -1.0), None),
    )
    assert predict_multi_pedestrian_positions(
        states, 2.0
    ) == predict_multi_pedestrian_positions(states, 2.0)


def test_multi_prediction_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        predict_multi_pedestrian_positions((), -0.1)


# ---------------------------------------------------------------------------
# Part F: space-time planning
# ---------------------------------------------------------------------------

def _two_humans():
    return (
        ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0)),
        ((3.0, 0.0), (-1.0, 0.0), None),
    )


def test_space_time_action_collision_with_second_human_rejects_action() -> None:
    safe_for_first_only = is_space_time_action_safe(
        (0, 0),
        (1, 0),
        0,
        pedestrian_position=(0.0, 0.0),
        pedestrian_velocity=(1.0, 0.0),
        pedestrian_target=(3.0, 0.0),
        move_duration=1.0,
        grid_scale=1.0,
        collision_distance=0.34,
        additional_pedestrians=(((3.0, 0.0), (-1.0, 0.0), None),),
    )

    assert not safe_for_first_only


def test_space_time_wait_checks_all_humans() -> None:
    # WAIT keeps the robot at (0,0); the second human crosses through it.
    assert not is_space_time_action_safe(
        (0, 0),
        (0, 0),
        0,
        pedestrian_position=(5.0, 0.0),
        pedestrian_velocity=(0.0, 0.0),
        pedestrian_target=(5.0, 0.0),
        move_duration=1.0,
        grid_scale=1.0,
        collision_distance=0.34,
        additional_pedestrians=(
            ((2.0, 0.0), (-2.0, 0.0), None),
        ),
    )


def test_time_aligned_social_cost_sums_humans() -> None:
    single = compute_time_aligned_social_cost(
        (0, 0),
        0,
        pedestrian_position=(0.2, 0.0),
        pedestrian_velocity=(0.0, 0.0),
        pedestrian_target=(0.2, 0.0),
        move_duration=1.0,
        grid_scale=1.0,
        social_distance=0.7,
        social_weight=10.0,
    )

    doubled = compute_time_aligned_social_cost(
        (0, 0),
        0,
        pedestrian_position=(0.2, 0.0),
        pedestrian_velocity=(0.0, 0.0),
        pedestrian_target=(0.2, 0.0),
        move_duration=1.0,
        grid_scale=1.0,
        social_distance=0.7,
        social_weight=10.0,
        additional_pedestrians=(
            ((0.2, 0.0), (0.0, 0.0), (0.2, 0.0)),
        ),
    )

    assert doubled == pytest.approx(2.0 * single)


def test_space_time_planning_is_deterministic_with_multiple_humans() -> None:
    def plan():
        return diagnose_space_time_social_astar(
            GridMap(3, 1),
            (0, 0),
            (2, 0),
            (5.0, 5.0),
            (0.0, 0.0),
            (5.0, 5.0),
            0.7,
            10.0,
            0.34,
            1.0,
            1.0,
            10.0,
            additional_pedestrians=(
                ((5.0, 6.0), (0.0, 0.0), (5.0, 6.0)),
            ),
        )

    assert plan() == plan()


def test_space_time_n1_equivalence_between_primary_and_additional() -> None:
    arguments = (
        GridMap(3, 1),
        (0, 0),
        (2, 0),
        (1.0, 0.0),
        (-1.0, 0.0),
        None,
        0.7,
        10.0,
        0.34,
        1.0,
        1.0,
        10.0,
    )

    primary = diagnose_space_time_social_astar(*arguments)
    as_additional = diagnose_space_time_social_astar(
        *arguments[:3],
        None,
        (0.0, 0.0),
        None,
        *arguments[6:],
        additional_pedestrians=(
            (arguments[3], arguments[4], arguments[5]),
        ),
    )

    assert primary == as_additional


# ---------------------------------------------------------------------------
# Parts G/H: robust bridge and egress
# ---------------------------------------------------------------------------

def test_bridge_safe_for_a_but_unsafe_for_b_is_rejected() -> None:
    evaluations = build_continuous_start_transitions(
        GridMap(3, 1),
        (0.0, 0.0),
        (0, 0),
        pedestrian_position=(10.0, 10.0),
        pedestrian_velocity=(0.0, 0.0),
        pedestrian_target=(10.0, 10.0),
        grid_scale=1.0,
        robot_speed=1.0,
        collision_distance=0.34,
        additional_pedestrians=(
            ((1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
        ),
    )
    toward_b = next(
        evaluation
        for evaluation in evaluations
        if evaluation.target_cell == (1, 0)
    )

    assert not toward_b.safe
    bridge = select_continuous_start_bridge(evaluations, (0.0, 0.0))
    assert bridge is not None
    assert bridge.target_cell == (0, 0)


def test_bridge_minimum_separation_uses_all_humans() -> None:
    evaluations = build_continuous_start_transitions(
        GridMap(3, 1),
        (0.0, 0.0),
        (0, 0),
        pedestrian_position=(0.0, 10.0),
        pedestrian_velocity=(0.0, 0.0),
        pedestrian_target=(0.0, 10.0),
        grid_scale=1.0,
        robot_speed=1.0,
        collision_distance=0.34,
        additional_pedestrians=(
            ((0.0, 0.5), (0.0, 0.0), (0.0, 0.5)),
        ),
    )
    selected = select_continuous_start_bridge(evaluations, (0.0, 0.0))

    assert selected is not None
    assert selected.minimum_predicted_separation == pytest.approx(0.5)


def test_egress_toward_second_human_is_rejected() -> None:
    assert not is_multi_collision_egress_motion_safe(
        (0.2, 0.0),
        (1.0, 0.0),
        (
            ((0.0, 0.0), (0.2, 0.6, 1.0)),
            ((1.0, 0.0), (0.8, 0.1, 0.05)),
        ),
        0.34,
    )


def test_valid_multi_human_egress_is_accepted() -> None:
    assert is_multi_collision_egress_motion_safe(
        (0.2, 0.0),
        (1.0, 0.0),
        (
            ((0.0, 0.0), (0.2, 0.6, 1.0)),
            ((1.0, 3.0), (3.1, 2.5, 2.0)),
        ),
        0.34,
    )


# ---------------------------------------------------------------------------
# Part I: direction-aware execution
# ---------------------------------------------------------------------------

def test_escape_toward_any_critical_human_is_rejected() -> None:
    # Away from A (0.1, 0) but straight toward B (0.4, 0): rejected.
    scale = compute_directional_speed_scale(
        (0.0, 0.0),
        (0.1, 0.0),
        (1.0, 0.0),
        0.5,
        1.5,
        0.25,
        additional_pedestrian_positions=[(0.4, 0.0)],
    )

    assert scale == 0.0


def test_escape_away_from_all_critical_humans_is_allowed() -> None:
    scale = compute_directional_speed_scale(
        (0.0, 0.0),
        (-0.1, 0.0),
        (1.0, 0.0),
        0.5,
        1.5,
        0.25,
        additional_pedestrian_positions=[(-0.2, 0.4)],
    )

    assert scale == 0.25


def test_multi_speed_scale_uses_closest_human_and_matches_single() -> None:
    assert compute_multi_speed_scale(
        (0.0, 0.0),
        [(1.0, 0.0), (0.3, 0.0), (2.0, 2.0)],
        0.5,
        1.5,
    ) == compute_speed_scale((0.0, 0.0), (0.3, 0.0), 0.5, 1.5)
    assert compute_multi_speed_scale(
        (0.0, 0.0),
        [(1.0, 0.0)],
        0.5,
        1.5,
    ) == compute_speed_scale((0.0, 0.0), (1.0, 0.0), 0.5, 1.5)
    assert compute_multi_speed_scale((0.0, 0.0), [], 0.5, 1.5) == 1.0


# ---------------------------------------------------------------------------
# Part L: generator
# ---------------------------------------------------------------------------

def test_multi_generation_is_deterministic_for_same_seed_and_count() -> None:
    for count in (0, 1, 3, 5, 10):
        first = generate_diverse_scenarios(10, seed=42, pedestrian_count=count)
        second = generate_diverse_scenarios(
            10, seed=42, pedestrian_count=count
        )
        assert second == first


def test_multi_generation_supports_requested_counts() -> None:
    for count in (3, 5, 10):
        scenarios = generate_scenarios(5, seed=7, pedestrian_count=count)
        assert all(
            scenario.pedestrian_count == count for scenario in scenarios
        )


def test_multi_generation_has_no_duplicate_pedestrian_specs() -> None:
    for count in (3, 5, 10):
        for scenario in generate_diverse_scenarios(
            10, seed=42, pedestrian_count=count
        ):
            specs = scenario.pedestrians
            assert len(set(specs)) == len(specs)
            starts = [spec.start for spec in specs]
            assert len(set(starts)) == len(starts)


def test_multi_generation_endpoints_are_valid() -> None:
    for scenario in generate_diverse_scenarios(
        10, seed=42, pedestrian_count=5
    ):
        grid_map = build_scenario_grid(scenario)
        robot_start = (
            scenario.start[0] * scenario.grid_scale,
            scenario.start[1] * scenario.grid_scale,
        )
        for spec in scenario.pedestrians:
            start_cell = (
                round(spec.start[0] / scenario.grid_scale),
                round(spec.start[1] / scenario.grid_scale),
            )
            target_cell = (
                round(spec.target[0] / scenario.grid_scale),
                round(spec.target[1] / scenario.grid_scale),
            )
            assert grid_map.is_free(start_cell)
            assert grid_map.is_free(target_cell)
            assert start_cell != target_cell
            assert spec.speed > 0.0
            assert hypot(
                spec.start[0] - robot_start[0],
                spec.start[1] - robot_start[1],
            ) >= scenario.grid_scale


def test_multi_generation_is_planner_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("social/predictive/space-time/robust planner "
                             "must not select multi-human scenarios")

    monkeypatch.setattr(scenario_module, "social_astar", fail_if_called)
    for planner_name in (
        "predictive_social_astar",
        "space_time_social_astar",
        "robust_space_time_social_astar",
    ):
        monkeypatch.setattr(
            scenario_module,
            planner_name,
            fail_if_called,
            raising=False,
        )

    scenarios = generate_diverse_scenarios(
        10,
        seed=42,
        pedestrian_count=3,
    )

    assert len(scenarios) == 10
    assert all(
        scenario.pedestrian_count == 3 for scenario in scenarios
    )


def test_negative_pedestrian_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        generate_diverse_scenarios(3, seed=42, pedestrian_count=-1)


# ---------------------------------------------------------------------------
# Runner-level multi-human checks
# ---------------------------------------------------------------------------

def test_runner_supports_zero_pedestrians() -> None:
    scenario = Scenario(
        scenario_id="empty-population",
        grid_width=2,
        grid_height=1,
        obstacle_cells=(),
        start=(0, 0),
        goal=(1, 0),
        grid_scale=0.75,
        pedestrians=(),
    )

    result, trace = run_episode_with_trace(scenario, "astar")

    assert result.success
    assert result.human_collision is False
    assert result.minimum_human_distance is None
    assert result.social_violation_rate == 0.0
    assert trace.pedestrian_count == 0
    assert trace.final_pedestrian_positions == ()
    assert trace.per_human_minimum_distances == ()


def test_runner_supports_one_and_three_pedestrians() -> None:
    for count in (1, 3):
        scenario = generate_diverse_scenarios(
            1, seed=42, pedestrian_count=count
        )[0]
        result, trace = run_episode_with_trace(
            scenario,
            "dynamic",
            max_steps=10,
        )

        assert trace.pedestrian_count == count
        assert len(trace.initial_pedestrian_positions) == count
        assert len(trace.initial_pedestrian_velocities) == count
        assert len(trace.pedestrian_targets) == count
        assert len(trace.final_pedestrian_positions) == count
        assert len(trace.per_human_minimum_distances) == count
        if result.minimum_human_distance is not None:
            assert min(
                distance
                for distance in trace.per_human_minimum_distances
                if distance is not None
            ) == pytest.approx(result.minimum_human_distance)


def test_runner_repeated_multi_human_run_is_deterministic() -> None:
    scenario = generate_diverse_scenarios(
        1, seed=42, pedestrian_count=3
    )[0]

    first = run_episode_with_trace(
        scenario,
        "social_replan_escape",
        max_steps=20,
    )
    second = run_episode_with_trace(
        scenario,
        "social_replan_escape",
        max_steps=20,
    )

    assert second == first


def test_runner_multi_human_robust_method_executes() -> None:
    scenario = generate_diverse_scenarios(
        1, seed=42, pedestrian_count=3
    )[0]

    result, trace = run_episode_with_trace(
        scenario,
        "social_spacetime_robust",
        max_steps=30,
    )

    assert trace.pedestrian_count == 3
    assert trace.continuous_bridge_attempts == 1
    assert result.steps <= 30


def test_per_human_minimum_distances_align_with_trajectories() -> None:
    robot = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    humans = [
        [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)],
        [(0.0, 0.5), (1.0, 0.5), (2.0, 0.5)],
    ]

    assert compute_per_human_minimum_distances(robot, humans) == (
        1.0,
        0.5,
    )
    assert compute_per_human_social_violation_rates(
        robot, humans, 0.7
    ) == pytest.approx((0.0, 1.0))
    assert colliding_human_indices(robot, humans, 0.34) == ()