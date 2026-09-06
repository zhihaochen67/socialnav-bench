import pytest

import socialnav.benchmark.runner as runner_module
import socialnav.benchmark.space_time_runner as space_time_runner_module
from experiments.run_benchmark import build_parser
from socialnav.benchmark import (
    SUPPORTED_METHODS,
    Scenario,
    aggregate_results,
    generate_diverse_scenarios,
    generate_scenarios,
    run_episode,
    run_episode_with_trace,
)
from socialnav.env.world import SIMULATION_STEP, SLOW_DISTANCE
from socialnav.evaluation import EpisodeResult
from socialnav.planners import SpaceTimePlan


def _result(
    *,
    success: bool,
    path_length: float,
    time_to_goal: float | None,
    spl: float,
    minimum_human_distance: float | None,
    social_violation_rate: float,
    human_collision: bool = False,
    obstacle_collision: bool = False,
) -> EpisodeResult:
    return EpisodeResult(
        success=success,
        path_length=path_length,
        time_to_goal=time_to_goal,
        spl=spl,
        minimum_human_distance=minimum_human_distance,
        social_violation_rate=social_violation_rate,
        human_collision=human_collision,
        obstacle_collision=obstacle_collision,
        steps=10,
    )


def test_aggregation_computes_all_requested_rates_and_means() -> None:
    summary = aggregate_results(
        [
            _result(
                success=True,
                path_length=10.0,
                time_to_goal=5.0,
                spl=1.0,
                minimum_human_distance=1.0,
                social_violation_rate=0.1,
            ),
            _result(
                success=False,
                path_length=8.0,
                time_to_goal=None,
                spl=0.0,
                minimum_human_distance=0.5,
                social_violation_rate=0.5,
                human_collision=True,
            ),
            _result(
                success=True,
                path_length=12.0,
                time_to_goal=6.0,
                spl=0.8,
                minimum_human_distance=None,
                social_violation_rate=0.0,
                obstacle_collision=True,
            ),
        ]
    )

    assert summary.episodes == 3
    assert summary.success_rate == pytest.approx(2.0 / 3.0)
    assert summary.collision_rate == pytest.approx(2.0 / 3.0)
    assert summary.mean_path_length == 10.0
    assert summary.mean_time_to_goal == 5.5
    assert summary.mean_spl == pytest.approx(0.6)
    assert summary.mean_minimum_human_distance == 0.75
    assert summary.mean_social_violation_rate == pytest.approx(0.2)


def test_aggregation_handles_no_successes_or_defined_human_distances() -> None:
    summary = aggregate_results(
        [
            _result(
                success=False,
                path_length=1.0,
                time_to_goal=None,
                spl=0.0,
                minimum_human_distance=None,
                social_violation_rate=0.0,
            )
        ]
    )

    assert summary.mean_time_to_goal is None
    assert summary.mean_minimum_human_distance is None


def test_aggregation_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="results must not be empty"):
        aggregate_results([])


def test_runner_rejects_unsupported_method() -> None:
    scenario = generate_scenarios(1, seed=42)[0]

    with pytest.raises(ValueError, match="method must be one of"):
        run_episode(scenario, "unknown")


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_runner_completes_headless_episode_for_each_method(method: str) -> None:
    scenario = generate_scenarios(1, seed=42)[0]

    result = run_episode(scenario, method)

    assert result.success
    assert result.steps > 0
    assert result.time_to_goal is not None
    assert result.minimum_human_distance is not None
    assert not result.obstacle_collision


def test_runner_timeout_is_reported_as_failure() -> None:
    scenario = generate_scenarios(1, seed=42)[0]

    result = run_episode(scenario, "astar", max_steps=1)

    assert not result.success
    assert result.steps == 1
    assert result.time_to_goal is None
    assert result.spl == 0.0


def test_repeated_headless_episode_is_deterministic() -> None:
    scenario = generate_scenarios(1, seed=42)[0]

    first = run_episode(scenario, "dynamic")
    second = run_episode(scenario, "dynamic")

    assert second == first


def test_runner_rejects_nonpositive_timeout() -> None:
    scenario = generate_scenarios(1, seed=42)[0]

    with pytest.raises(ValueError, match="max_steps must be positive"):
        run_episode(scenario, "astar", max_steps=0)


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_runner_accepts_diverse_scenario(method: str) -> None:
    scenario = generate_diverse_scenarios(1, seed=42)[0]

    result = run_episode(scenario, method)

    assert result.steps > 0
    assert result.minimum_human_distance is not None


def test_cli_defaults_to_controlled_scenario_mode() -> None:
    args = build_parser().parse_args([])

    assert args.scenario_mode == "controlled"


@pytest.mark.parametrize("mode", ["controlled", "diverse"])
def test_cli_accepts_supported_scenario_modes(mode: str) -> None:
    args = build_parser().parse_args(["--scenario-mode", mode])

    assert args.scenario_mode == mode


def test_cli_rejects_unknown_scenario_mode() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--scenario-mode", "unknown"])


def _blocking_scenario() -> Scenario:
    return Scenario(
        scenario_id="online-replan-test",
        grid_width=3,
        grid_height=2,
        obstacle_cells=(),
        start=(0, 0),
        goal=(2, 0),
        grid_scale=1.0,
        pedestrian_start=(0.25, 0.0),
        pedestrian_target=(0.25, 1.0),
        pedestrian_speed=1.0,
    )


def test_supported_method_order_includes_space_time_methods_last() -> None:
    assert SUPPORTED_METHODS == (
        "astar",
        "dynamic",
        "social",
        "social_replan",
        "social_replan_escape",
        "social_replan_recovery",
        "social_predictive",
        "social_predictive_replan",
        "social_spacetime",
        "social_spacetime_replan",
        "social_spacetime_robust",
    )


@pytest.mark.parametrize(
    "method",
    ("astar", "dynamic", "social", "social_replan", "social_predictive"),
)
def test_existing_methods_never_record_replans(method: str) -> None:
    scenario = generate_scenarios(1, seed=42)[0]

    _, trace = run_episode_with_trace(scenario, method, max_steps=1)

    assert trace.replan_count == 0
    assert trace.replan_steps == ()
    assert trace.successful_replans == 0
    assert trace.failed_replans == 0


def test_social_replan_is_deterministic_on_repeated_runs() -> None:
    scenario = generate_diverse_scenarios(1, seed=42)[0]

    first = run_episode_with_trace(scenario, "social_replan")
    second = run_episode_with_trace(scenario, "social_replan")

    assert second == first


def test_replan_uses_current_pedestrian_and_replaces_stale_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, int], tuple[tuple[float, float], ...]]] = []

    def fake_social_astar(
        _grid_map: object,
        start: tuple[int, int],
        _goal: tuple[int, int],
        *,
        pedestrian_positions: list[tuple[float, float]],
        **_kwargs: object,
    ) -> list[tuple[int, int]]:
        calls.append((start, tuple(pedestrian_positions)))
        if len(calls) == 1:
            return [(0, 0), (1, 0), (2, 0)]
        return [start, (0, 1), (1, 1), (2, 1), (2, 0)]

    monkeypatch.setattr(runner_module, "social_astar", fake_social_astar)

    _, trace = run_episode_with_trace(
        _blocking_scenario(),
        "social_replan",
        max_steps=2,
        replan_stop_steps=2,
    )

    assert len(calls) == 2
    assert calls[0][1] == ((0.25, 0.0),)
    assert calls[1][0] == (0, 0)
    assert calls[1][1][0] == pytest.approx((0.25, SIMULATION_STEP))
    assert trace.planned_path == (
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 0.0),
    )
    assert trace.replan_count == 1
    assert trace.replan_steps == (2,)
    assert trace.successful_replans == 1
    assert trace.failed_replans == 0


def test_failed_replan_retains_path_and_requires_fresh_stop_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_social_astar(
        _grid_map: object,
        _start: tuple[int, int],
        _goal: tuple[int, int],
        **_kwargs: object,
    ) -> list[tuple[int, int]] | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [(0, 0), (1, 0), (2, 0)]
        return None

    monkeypatch.setattr(runner_module, "social_astar", fake_social_astar)

    result, trace = run_episode_with_trace(
        _blocking_scenario(),
        "social_replan",
        max_steps=4,
        replan_stop_steps=2,
    )

    assert result.steps == 4
    assert trace.planned_path == (
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    )
    assert trace.replan_count == 2
    assert trace.replan_steps == (2, 4)
    assert trace.successful_replans == 0
    assert trace.failed_replans == 2


@pytest.mark.parametrize(
    "method",
    (
        "social_replan",
        "social_replan_escape",
        "social_replan_recovery",
        "social_predictive_replan",
        "social_spacetime_replan",
    ),
)
def test_replanning_methods_reject_nonpositive_stop_threshold(method: str) -> None:
    with pytest.raises(ValueError, match="replan_stop_steps must be positive"):
        run_episode(
            _blocking_scenario(),
            method,
            replan_stop_steps=0,
        )


@pytest.mark.parametrize(
    "method",
    ("astar", "dynamic", "social", "social_replan"),
)
def test_existing_methods_do_not_use_directional_controller(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenarios(1, seed=42)[0]
    expected = run_episode_with_trace(scenario, method, max_steps=1)

    def unexpected_directional_call(*_args: object, **_kwargs: object) -> float:
        pytest.fail("existing method called the directional controller")

    monkeypatch.setattr(
        runner_module,
        "compute_directional_speed_scale",
        unexpected_directional_call,
    )

    actual = run_episode_with_trace(scenario, method, max_steps=1)

    assert actual == expected


def test_social_replan_escape_is_deterministic_on_repeated_runs() -> None:
    scenario = generate_diverse_scenarios(1, seed=42)[0]

    first = run_episode_with_trace(scenario, "social_replan_escape")
    second = run_episode_with_trace(scenario, "social_replan_escape")

    assert second == first


def _escape_blocking_scenario() -> Scenario:
    return Scenario(
        scenario_id="directional-escape-test",
        grid_width=3,
        grid_height=3,
        obstacle_cells=(),
        start=(1, 1),
        goal=(2, 1),
        grid_scale=1.0,
        pedestrian_start=(1.25, 1.0),
        pedestrian_target=(1.25, 1.0),
        pedestrian_speed=0.0,
    )


def test_escape_method_physically_moves_away_after_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_social_astar(
        _grid_map: object,
        start: tuple[int, int],
        _goal: tuple[int, int],
        **_kwargs: object,
    ) -> list[tuple[int, int]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [(1, 1), (2, 1)]
        return [start, (0, 1), (0, 0), (1, 0), (2, 0), (2, 1)]

    monkeypatch.setattr(runner_module, "social_astar", fake_social_astar)
    scenario = _escape_blocking_scenario()

    _, baseline_trace = run_episode_with_trace(
        scenario,
        "social_replan",
        max_steps=3,
        replan_stop_steps=2,
    )
    assert calls == 2

    calls = 0
    result, escape_trace = run_episode_with_trace(
        scenario,
        "social_replan_escape",
        max_steps=3,
        replan_stop_steps=2,
    )

    assert calls == 2
    assert result.steps == 3
    assert baseline_trace.speed_scales == (0.0, 0.0, 0.0)
    assert escape_trace.speed_scales == (0.0, 0.0, 0.25)
    assert baseline_trace.final_robot_position == pytest.approx((1.0, 1.0))
    assert escape_trace.final_robot_position[0] < 1.0
    assert escape_trace.final_robot_position[1] == pytest.approx(1.0)
    assert escape_trace.replan_count == 1
    assert escape_trace.replan_steps == (2,)
    assert escape_trace.successful_replans == 1
    assert escape_trace.failed_replans == 0

@pytest.mark.parametrize(
    "method",
    (
        "astar",
        "dynamic",
        "social",
        "social_replan",
        "social_replan_escape",
        "social_predictive",
        "social_predictive_replan",
    ),
)
def test_existing_methods_never_record_recoveries(method: str) -> None:
    scenario = generate_scenarios(1, seed=42)[0]

    _, trace = run_episode_with_trace(scenario, method, max_steps=1)

    assert trace.recovery_count == 0
    assert trace.recovery_trigger_steps == ()
    assert trace.successful_recoveries == 0
    assert trace.failed_recoveries == 0
    assert trace.recovery_path_lengths == ()


def test_recovery_not_used_when_replanned_route_can_start_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_social_astar(
        _grid_map: object,
        start: tuple[int, int],
        _goal: tuple[int, int],
        **_kwargs: object,
    ) -> list[tuple[int, int]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [(1, 1), (2, 1)]
        return [start, (0, 1), (0, 0), (1, 0), (2, 0), (2, 1)]

    def unexpected_recovery(*_args: object, **_kwargs: object) -> None:
        pytest.fail("safe replanned route triggered recovery")

    monkeypatch.setattr(runner_module, "social_astar", fake_social_astar)
    monkeypatch.setattr(
        runner_module,
        "find_clearance_recovery_path",
        unexpected_recovery,
    )

    _, trace = run_episode_with_trace(
        _escape_blocking_scenario(),
        "social_replan_recovery",
        max_steps=3,
        replan_stop_steps=2,
    )

    assert calls == 2
    assert trace.speed_scales == (0.0, 0.0, 0.25)
    assert trace.recovery_count == 0


def test_recovery_activates_moves_and_replans_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    social_calls: list[
        tuple[tuple[int, int], tuple[tuple[float, float], ...]]
    ] = []
    recovery_calls: list[
        tuple[
            tuple[int, int],
            tuple[float, float],
            tuple[float, float],
            float,
        ]
    ] = []

    def fake_social_astar(
        _grid_map: object,
        start: tuple[int, int],
        _goal: tuple[int, int],
        *,
        pedestrian_positions: list[tuple[float, float]],
        **_kwargs: object,
    ) -> list[tuple[int, int]]:
        social_calls.append((start, tuple(pedestrian_positions)))
        if len(social_calls) <= 2:
            return [start, (2, 1)]
        return [start, (0, 0), (1, 0), (2, 0), (2, 1)]

    def fake_recovery(
        _grid_map: object,
        start: tuple[int, int],
        robot_position: tuple[float, float],
        pedestrian_position: tuple[float, float],
        _grid_scale: float,
        target_clearance: float,
    ) -> list[tuple[int, int]]:
        recovery_calls.append(
            (
                start,
                robot_position,
                pedestrian_position,
                target_clearance,
            )
        )
        return [(0, 1)]

    monkeypatch.setattr(runner_module, "social_astar", fake_social_astar)
    monkeypatch.setattr(
        runner_module,
        "find_clearance_recovery_path",
        fake_recovery,
    )
    monkeypatch.setattr(runner_module, "STEPS_PER_CELL", 1)

    result, trace = run_episode_with_trace(
        _escape_blocking_scenario(),
        "social_replan_recovery",
        max_steps=7,
        replan_stop_steps=2,
    )

    assert result.steps == 7
    assert len(social_calls) == 3
    assert social_calls[2][0] == (0, 1)
    assert recovery_calls == [
        ((1, 1), (1.0, 1.0), (1.25, 1.0), SLOW_DISTANCE)
    ]
    assert trace.replan_count == 2
    assert trace.replan_steps == (2, 7)
    assert trace.successful_replans == 2
    assert trace.failed_replans == 0
    assert trace.recovery_count == 1
    assert trace.recovery_trigger_steps == (2,)
    assert trace.successful_recoveries == 1
    assert trace.failed_recoveries == 0
    assert trace.recovery_path_lengths == (1,)
    assert trace.final_robot_position[0] == pytest.approx(0.0)


def test_failed_recovery_is_recorded_and_waiting_remains_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_social_astar(
        _grid_map: object,
        start: tuple[int, int],
        _goal: tuple[int, int],
        **_kwargs: object,
    ) -> list[tuple[int, int]]:
        return [start, (2, 1)]

    monkeypatch.setattr(runner_module, "social_astar", fake_social_astar)
    monkeypatch.setattr(
        runner_module,
        "find_clearance_recovery_path",
        lambda *_args, **_kwargs: None,
    )

    result, trace = run_episode_with_trace(
        _escape_blocking_scenario(),
        "social_replan_recovery",
        max_steps=4,
        replan_stop_steps=2,
    )

    assert result.steps == 4
    assert trace.speed_scales == (0.0, 0.0, 0.0, 0.0)
    assert trace.replan_count == 2
    assert trace.replan_steps == (2, 4)
    assert trace.successful_replans == 2
    assert trace.failed_replans == 0
    assert trace.recovery_count == 2
    assert trace.recovery_trigger_steps == (2, 4)
    assert trace.successful_recoveries == 0
    assert trace.failed_recoveries == 2
    assert trace.recovery_path_lengths == ()
    assert trace.final_robot_position == pytest.approx((1.0, 1.0))


def test_social_replan_recovery_is_deterministic_on_repeated_runs() -> None:
    scenario = generate_diverse_scenarios(1, seed=42)[0]

    first = run_episode_with_trace(scenario, "social_replan_recovery")
    second = run_episode_with_trace(scenario, "social_replan_recovery")

    assert second == first


@pytest.mark.parametrize(
    "method",
    (
        "astar",
        "dynamic",
        "social",
        "social_replan",
        "social_replan_escape",
        "social_replan_recovery",
    ),
)
def test_existing_methods_do_not_use_predictive_planner(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenarios(1, seed=42)[0]
    expected = run_episode_with_trace(scenario, method, max_steps=1)

    def unexpected_predictive_call(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        pytest.fail("existing method called predictive social A*")

    monkeypatch.setattr(
        runner_module,
        "predictive_social_astar",
        unexpected_predictive_call,
    )

    actual = run_episode_with_trace(scenario, method, max_steps=1)

    assert actual == expected


def test_predictive_social_uses_initial_velocity_without_replanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[
        tuple[
            tuple[int, int],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
    ] = []

    def fake_predictive_social_astar(
        _grid_map: object,
        start: tuple[int, int],
        _goal: tuple[int, int],
        pedestrian_position: tuple[float, float],
        pedestrian_velocity: tuple[float, float],
        *_args: object,
        pedestrian_target: tuple[float, float],
        **_kwargs: object,
    ) -> list[tuple[int, int]]:
        calls.append(
            (
                start,
                pedestrian_position,
                pedestrian_velocity,
                pedestrian_target,
            )
        )
        return [(0, 0), (1, 0), (2, 0)]

    monkeypatch.setattr(
        runner_module,
        "predictive_social_astar",
        fake_predictive_social_astar,
    )

    _, trace = run_episode_with_trace(
        _blocking_scenario(),
        "social_predictive",
        max_steps=2,
        replan_stop_steps=1,
    )

    assert calls == [((0, 0), (0.25, 0.0), (0.0, 1.0), (0.25, 1.0))]
    assert trace.replan_count == 0


def test_predictive_replan_uses_current_position_velocity_and_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[
        tuple[
            tuple[int, int],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
    ] = []

    def fake_predictive_social_astar(
        _grid_map: object,
        start: tuple[int, int],
        _goal: tuple[int, int],
        pedestrian_position: tuple[float, float],
        pedestrian_velocity: tuple[float, float],
        *_args: object,
        pedestrian_target: tuple[float, float],
        **_kwargs: object,
    ) -> list[tuple[int, int]]:
        calls.append(
            (
                start,
                pedestrian_position,
                pedestrian_velocity,
                pedestrian_target,
            )
        )
        if len(calls) == 1:
            return [(0, 0), (1, 0), (2, 0)]
        return [start, (0, 1), (1, 1), (2, 1), (2, 0)]

    monkeypatch.setattr(
        runner_module,
        "predictive_social_astar",
        fake_predictive_social_astar,
    )

    _, trace = run_episode_with_trace(
        _blocking_scenario(),
        "social_predictive_replan",
        max_steps=2,
        replan_stop_steps=2,
    )

    assert len(calls) == 2
    assert calls[0] == (
        (0, 0),
        (0.25, 0.0),
        (0.0, 1.0),
        (0.25, 1.0),
    )
    assert calls[1][0] == (0, 0)
    assert calls[1][1] == pytest.approx((0.25, SIMULATION_STEP))
    assert calls[1][2] == pytest.approx((0.0, 1.0))
    assert calls[1][3] == (0.25, 1.0)
    assert trace.planned_path == (
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 0.0),
    )
    assert trace.replan_count == 1
    assert trace.replan_steps == (2,)
    assert trace.successful_replans == 1
    assert trace.failed_replans == 0


def test_failed_predictive_replan_retains_path_and_retries_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_predictive_social_astar(
        _grid_map: object,
        _start: tuple[int, int],
        _goal: tuple[int, int],
        *_args: object,
        **_kwargs: object,
    ) -> list[tuple[int, int]] | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [(0, 0), (1, 0), (2, 0)]
        return None

    monkeypatch.setattr(
        runner_module,
        "predictive_social_astar",
        fake_predictive_social_astar,
    )

    _, trace = run_episode_with_trace(
        _blocking_scenario(),
        "social_predictive_replan",
        max_steps=4,
        replan_stop_steps=2,
    )

    assert calls == 3
    assert trace.replan_count == 2
    assert trace.replan_steps == (2, 4)
    assert trace.successful_replans == 0
    assert trace.failed_replans == 2
    assert trace.planned_path == ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))


@pytest.mark.parametrize(
    "method",
    ("social_predictive", "social_predictive_replan"),
)
def test_predictive_methods_are_deterministic(method: str) -> None:
    scenario = generate_diverse_scenarios(1, seed=42)[0]

    first = run_episode_with_trace(scenario, method)
    second = run_episode_with_trace(scenario, method)

    assert second == first


def _space_time_runner_scenario(
    *,
    pedestrian_start: tuple[float, float] = (10.0, 10.0),
    pedestrian_target: tuple[float, float] = (10.0, 10.0),
    pedestrian_speed: float = 0.0,
) -> Scenario:
    return Scenario(
        scenario_id="space-time-runner-test",
        grid_width=2,
        grid_height=1,
        obstacle_cells=(),
        start=(0, 0),
        goal=(1, 0),
        grid_scale=0.75,
        pedestrian_start=pedestrian_start,
        pedestrian_target=pedestrian_target,
        pedestrian_speed=pedestrian_speed,
    )


def _space_time_plan(
    timed_states: tuple[tuple[int, int, int], ...],
    actions: tuple[str, ...],
) -> SpaceTimePlan:
    wait_actions = actions.count("WAIT")
    return SpaceTimePlan(
        spatial_path=tuple((state[0], state[1]) for state in timed_states),
        timed_states=timed_states,
        actions=actions,
        planned_wait_actions=wait_actions,
        planned_move_actions=len(actions) - wait_actions,
        move_duration=0.375,
        estimated_duration=len(actions) * 0.375,
    )


@pytest.mark.parametrize(
    "method",
    (
        "astar",
        "dynamic",
        "social",
        "social_replan",
        "social_replan_escape",
        "social_replan_recovery",
        "social_predictive",
        "social_predictive_replan",
    ),
)
def test_existing_eight_methods_do_not_dispatch_to_space_time_runner(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenarios(1, seed=42)[0]
    expected = run_episode_with_trace(scenario, method, max_steps=1)

    def unexpected_space_time_call(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        pytest.fail("existing method dispatched to space-time runner")

    monkeypatch.setattr(
        runner_module,
        "run_space_time_episode_with_trace",
        unexpected_space_time_call,
    )

    actual = run_episode_with_trace(scenario, method, max_steps=1)

    assert actual == expected


def test_planned_wait_executes_for_exact_duration_without_reactive_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _space_time_plan(
        ((0, 0, 0), (0, 0, 1), (1, 0, 2)),
        ("WAIT", "RIGHT"),
    )
    monkeypatch.setattr(
        space_time_runner_module,
        "space_time_social_astar",
        lambda *_args, **_kwargs: plan,
    )

    result, trace = run_episode_with_trace(
        _space_time_runner_scenario(),
        "social_spacetime_replan",
        max_steps=90,
        replan_stop_steps=1,
    )

    assert result.steps == 90
    assert trace.planned_wait_actions == 1
    assert trace.executed_wait_actions == 1
    assert trace.planned_move_actions == 1
    assert trace.total_intentional_wait_steps == 90
    assert trace.reactive_stopped_steps == 0
    assert trace.replan_count == 0
    assert trace.speed_scales == (1.0,) * 90


def test_reactive_block_uses_current_pedestrian_state_and_triggers_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[
        tuple[
            tuple[int, int],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float] | None,
        ]
    ] = []
    plan = _space_time_plan(
        ((0, 0, 0), (1, 0, 1)),
        ("RIGHT",),
    )

    def fake_space_time_social_astar(
        _grid_map: object,
        start: tuple[int, int],
        _goal: tuple[int, int],
        pedestrian_position: tuple[float, float],
        pedestrian_velocity: tuple[float, float],
        pedestrian_target: tuple[float, float] | None,
        *_args: object,
        **_kwargs: object,
    ) -> SpaceTimePlan:
        calls.append(
            (
                start,
                pedestrian_position,
                pedestrian_velocity,
                pedestrian_target,
            )
        )
        return plan

    monkeypatch.setattr(
        space_time_runner_module,
        "space_time_social_astar",
        fake_space_time_social_astar,
    )
    scenario = _space_time_runner_scenario(
        pedestrian_start=(0.25, 0.0),
        pedestrian_target=(0.25, 1.0),
        pedestrian_speed=1.0,
    )

    _, trace = run_episode_with_trace(
        scenario,
        "social_spacetime_replan",
        max_steps=2,
        replan_stop_steps=2,
    )

    assert len(calls) == 2
    assert calls[0] == (
        (0, 0),
        (0.25, 0.0),
        (0.0, 1.0),
        (0.25, 1.0),
    )
    assert calls[1][0] == (0, 0)
    assert calls[1][1] == pytest.approx((0.25, SIMULATION_STEP))
    assert calls[1][2] == pytest.approx((0.0, 1.0))
    assert calls[1][3] == (0.25, 1.0)
    assert trace.replan_count == 1
    assert trace.replan_steps == (2,)
    assert trace.successful_replans == 1
    assert trace.failed_replans == 0
    assert trace.spacetime_plan_count == 2
    assert trace.spacetime_planning_failures == 0
    assert trace.reactive_stopped_steps == 2
    assert trace.total_intentional_wait_steps == 0


def test_failed_initial_space_time_plan_is_safe_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        space_time_runner_module,
        "space_time_social_astar",
        lambda *_args, **_kwargs: None,
    )

    result, trace = run_episode_with_trace(
        _space_time_runner_scenario(),
        "social_spacetime",
        max_steps=3,
    )

    assert not result.success
    assert result.steps == 3
    assert trace.timed_out
    assert trace.spacetime_plan_count == 1
    assert trace.spacetime_planning_failures == 1
    assert trace.planned_wait_actions == 0
    assert trace.executed_wait_actions == 0
    assert trace.total_intentional_wait_steps == 0
    assert trace.reactive_stopped_steps == 0
    assert trace.speed_scales == (1.0, 1.0, 1.0)


def test_failed_space_time_replan_retains_plan_and_retries_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    plan = _space_time_plan(
        ((0, 0, 0), (1, 0, 1)),
        ("RIGHT",),
    )

    def fake_space_time_social_astar(
        *_args: object,
        **_kwargs: object,
    ) -> SpaceTimePlan | None:
        nonlocal calls
        calls += 1
        return plan if calls == 1 else None

    monkeypatch.setattr(
        space_time_runner_module,
        "space_time_social_astar",
        fake_space_time_social_astar,
    )

    _, trace = run_episode_with_trace(
        _space_time_runner_scenario(
            pedestrian_start=(0.25, 0.0),
            pedestrian_target=(0.25, 0.0),
        ),
        "social_spacetime_replan",
        max_steps=4,
        replan_stop_steps=2,
    )

    assert calls == 3
    assert trace.planned_path == ((0.0, 0.0), (0.75, 0.0))
    assert trace.replan_count == 2
    assert trace.replan_steps == (2, 4)
    assert trace.successful_replans == 0
    assert trace.failed_replans == 2
    assert trace.spacetime_plan_count == 3
    assert trace.spacetime_planning_failures == 2
    assert trace.reactive_stopped_steps == 4


@pytest.mark.parametrize(
    "method",
    ("social_spacetime", "social_spacetime_replan"),
)
def test_space_time_methods_are_deterministic(method: str) -> None:
    scenario = generate_diverse_scenarios(1, seed=42)[0]

    first = run_episode_with_trace(scenario, method)
    second = run_episode_with_trace(scenario, method)

    assert second == first
