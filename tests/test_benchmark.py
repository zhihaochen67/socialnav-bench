import pytest

from socialnav.benchmark import (
    aggregate_results,
    generate_scenarios,
    run_episode,
)
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


@pytest.mark.parametrize("method", ["astar", "dynamic", "social"])
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
