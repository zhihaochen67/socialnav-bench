from dataclasses import replace

import pytest

from experiments.analyze_failures import _summarize, build_parser
from socialnav.benchmark import (
    PATH_NEAR_THRESHOLD,
    EpisodeTrace,
    Scenario,
    classify_failure,
    count_stopped_steps,
    diagnose_failure,
    did_episode_time_out,
    generate_scenarios,
    is_point_on_or_near_route,
    late_stopped_fraction,
    longest_stopped_streak,
    nearest_route_waypoint,
    point_to_route_distance,
    run_episode_with_trace,
    stopped_fraction,
)
from socialnav.evaluation import EpisodeResult


def _classification_inputs() -> dict[str, object]:
    return {
        "success": False,
        "timed_out": True,
        "human_collision": False,
        "obstacle_collision": False,
        "pedestrian_near_path": True,
        "robot_to_pedestrian_distance": 0.5,
        "late_stop_fraction": 1.0,
        "longest_stop_steps": 25,
        "total_steps": 100,
    }


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="diagnostic-test",
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


def _failed_result() -> EpisodeResult:
    return EpisodeResult(
        success=False,
        path_length=0.5,
        time_to_goal=None,
        spl=0.0,
        minimum_human_distance=0.5,
        social_violation_rate=0.75,
        human_collision=False,
        obstacle_collision=False,
        steps=4,
    )


def _trace() -> EpisodeTrace:
    return EpisodeTrace(
        planned_path=((0.0, 0.0), (2.0, 0.0)),
        speed_scales=(0.0, 0.0, 0.0, 0.0),
        timed_out=True,
        final_robot_position=(0.5, 0.0),
        final_pedestrian_position=(1.0, 0.0),
        max_steps=4,
    )


def test_point_to_route_distance_uses_segments_in_world_space() -> None:
    route = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0))

    assert point_to_route_distance((1.0, 1.0), route) == 1.0
    assert point_to_route_distance((3.0, 2.0), route) == 1.0
    assert nearest_route_waypoint((1.8, 1.7), route) == (2.0, 2.0)


def test_on_or_near_route_includes_exact_threshold() -> None:
    route = ((0.0, 0.0), (2.0, 0.0))

    assert is_point_on_or_near_route(
        (1.0, PATH_NEAR_THRESHOLD),
        route,
    )
    assert not is_point_on_or_near_route(
        (1.0, PATH_NEAR_THRESHOLD + 1e-6),
        route,
    )


def test_stopped_step_counts_fractions_and_longest_streak() -> None:
    speed_scales = (0.0, 1e-13, 0.5, 0.0, 0.0, 1.0)

    assert count_stopped_steps(speed_scales) == 4
    assert stopped_fraction(speed_scales) == pytest.approx(4.0 / 6.0)
    assert longest_stopped_streak(speed_scales) == 2
    assert late_stopped_fraction(speed_scales, late_fraction=0.5) == (
        pytest.approx(2.0 / 3.0)
    )


def test_timeout_requires_exhausted_budget_and_incomplete_path() -> None:
    assert did_episode_time_out(
        steps=100,
        max_steps=100,
        path_completed=False,
    )
    assert not did_episode_time_out(
        steps=100,
        max_steps=100,
        path_completed=True,
    )
    assert not did_episode_time_out(
        steps=99,
        max_steps=100,
        path_completed=False,
    )


def test_pedestrian_blocking_classification_is_deterministic() -> None:
    inputs = _classification_inputs()

    assert classify_failure(**inputs) == "pedestrian_blocking_path"
    assert classify_failure(**inputs) == "pedestrian_blocking_path"


def test_collision_classification_has_precedence() -> None:
    inputs = _classification_inputs()
    inputs["human_collision"] = True

    assert classify_failure(**inputs) == "collision"


def test_reactive_wait_timeout_when_waiting_is_not_path_blocking() -> None:
    inputs = _classification_inputs()
    inputs["pedestrian_near_path"] = False
    inputs["robot_to_pedestrian_distance"] = 2.0

    assert classify_failure(**inputs) == "reactive_wait_timeout"


def test_goal_not_reached_and_other_fallbacks() -> None:
    inputs = _classification_inputs()
    inputs["timed_out"] = False
    inputs["late_stop_fraction"] = 0.0
    inputs["longest_stop_steps"] = 0
    assert classify_failure(**inputs) == "goal_not_reached"

    inputs["timed_out"] = True
    assert classify_failure(**inputs) == "other"


def test_successful_episode_is_not_classified_or_diagnosed() -> None:
    inputs = _classification_inputs()
    inputs["success"] = True
    assert classify_failure(**inputs) is None

    successful = replace(
        _failed_result(),
        success=True,
        time_to_goal=1.0,
        spl=1.0,
    )
    assert diagnose_failure(
        _scenario(),
        "social",
        successful,
        _trace(),
    ) is None


def test_repeated_failure_diagnostics_are_identical() -> None:
    first = diagnose_failure(
        _scenario(),
        "social",
        _failed_result(),
        _trace(),
    )
    second = diagnose_failure(
        _scenario(),
        "social",
        _failed_result(),
        _trace(),
    )

    assert first is not None
    assert second == first
    assert first.likely_failure_reason == "pedestrian_blocking_path"
    assert first.pedestrian_final_position_on_or_near_path
    assert first.robot_stopped_steps == 4
    assert first.longest_consecutive_stop_steps == 4


def test_summary_keeps_categories_exclusive_but_counts_wait_evidence() -> None:
    diagnostic = diagnose_failure(
        _scenario(),
        "social",
        _failed_result(),
        _trace(),
    )
    assert diagnostic is not None
    collision = replace(
        diagnostic,
        human_collision=True,
        likely_failure_reason="collision",
    )

    summary = _summarize(1, [collision])

    assert summary["counts_by_reason"]["collision"] == 1
    assert summary["wait_timeout_failures"] == 1
    assert summary["pedestrian_blocking_evidence"] == 1


def test_runner_trace_records_one_speed_scale_per_step() -> None:
    scenario = generate_scenarios(1, seed=42)[0]

    result, trace = run_episode_with_trace(
        scenario,
        "astar",
        max_steps=1,
    )

    assert result.steps == 1
    assert trace.speed_scales == (1.0,)
    assert trace.timed_out
    assert trace.max_steps == 1
    assert trace.final_robot_position
    assert trace.final_pedestrian_position
    assert trace.planned_path


def test_failure_analysis_cli_defaults_to_requested_configuration() -> None:
    args = build_parser().parse_args([])

    assert args.episodes == 100
    assert args.seed == 42
    assert args.scenario_mode == "diverse"
