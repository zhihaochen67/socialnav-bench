import pytest

import socialnav.benchmark.runner as runner_module
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
from socialnav.env.world import SIMULATION_STEP
from socialnav.evaluation import EpisodeResult


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


def test_supported_method_order_includes_social_replan_last() -> None:
    assert SUPPORTED_METHODS == (
        "astar",
        "dynamic",
        "social",
        "social_replan",
    )


@pytest.mark.parametrize("method", ("astar", "dynamic", "social"))
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


def test_social_replan_rejects_nonpositive_stop_threshold() -> None:
    with pytest.raises(ValueError, match="replan_stop_steps must be positive"):
        run_episode(
            _blocking_scenario(),
            "social_replan",
            replan_stop_steps=0,
        )
