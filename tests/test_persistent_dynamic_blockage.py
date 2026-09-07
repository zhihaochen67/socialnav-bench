"""Focused tests for read-only persistent dynamic blockage diagnostics."""

from socialnav.benchmark import PedestrianSpec, Scenario
from socialnav.benchmark.diagnostics import EpisodeTrace
from socialnav.benchmark.failure_analysis import (
    classify_robust_failure,
    diagnose_persistent_dynamic_blockage,
)
from socialnav.benchmark.robust_reporting import summarize_robust_execution
from socialnav.evaluation import EpisodeResult


def _scenario(
    *,
    height: int = 1,
    obstacles: tuple[tuple[int, int], ...] = (),
    pedestrians: tuple[PedestrianSpec, ...] = (),
) -> Scenario:
    return Scenario(
        scenario_id="persistent-blockage-test",
        grid_width=3,
        grid_height=height,
        obstacle_cells=obstacles,
        start=(0, 0),
        goal=(2, 0),
        grid_scale=1.0,
        pedestrians=pedestrians,
    )


def _trace(
    final_pedestrian_positions: tuple[tuple[float, float], ...],
    *,
    planner_failure_reason: str | None = "time_horizon_exhausted",
    episode_failure_reason: str | None = "post_override_replan_failure",
) -> EpisodeTrace:
    return EpisodeTrace(
        planned_path=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
        speed_scales=(0.0,),
        timed_out=True,
        final_robot_position=(0.0, 0.0),
        final_pedestrian_position=(
            final_pedestrian_positions[0]
            if final_pedestrian_positions
            else (99.0, 99.0)
        ),
        max_steps=10,
        robust_planning_failure_reasons=(
            (planner_failure_reason,)
            if planner_failure_reason is not None
            else ()
        ),
        robust_episode_failure_reason=episode_failure_reason,
        pedestrian_count=len(final_pedestrian_positions),
        final_pedestrian_positions=final_pedestrian_positions,
    )


def _failure_result() -> EpisodeResult:
    return EpisodeResult(
        success=False,
        path_length=0.0,
        time_to_goal=None,
        spl=0.0,
        minimum_human_distance=1.0,
        social_violation_rate=0.0,
        human_collision=False,
        obstacle_collision=False,
        steps=10,
    )


def _terminal_pedestrian(position: tuple[float, float]) -> PedestrianSpec:
    return PedestrianSpec(start=position, target=position, speed=0.5)


def test_single_terminal_stationary_pedestrian_blocks_corridor() -> None:
    scenario = _scenario(
        pedestrians=(_terminal_pedestrian((1.0, 0.0)),),
    )
    trace = _trace(((1.0, 0.0),))

    diagnosis = diagnose_persistent_dynamic_blockage(scenario, trace)

    assert diagnosis.static_connectivity
    assert not diagnosis.connectivity_with_persistent_blockers
    assert diagnosis.removing_dynamic_blockers_restores_static_connectivity
    assert diagnosis.persistent_dynamic_blockage
    assert diagnosis.diagnostic_failure_category == "persistent_dynamic_blockage"
    assert diagnosis.persistent_blockage_type == "terminal_pedestrian_blockage"
    assert diagnosis.blocking_pedestrian_ids == (0,)
    assert diagnosis.blocking_cells == ((1, 0),)
    assert diagnosis.blocker_count == 1
    assert not diagnosis.multiple_humans_jointly_form_cut
    assert diagnosis.pedestrian_evidence[0].terminal
    assert diagnosis.pedestrian_evidence[0].stationary


def test_two_terminal_pedestrians_jointly_form_minimum_cut() -> None:
    scenario = _scenario(
        height=2,
        pedestrians=(
            _terminal_pedestrian((1.0, 0.0)),
            _terminal_pedestrian((1.0, 1.0)),
        ),
    )
    trace = _trace(((1.0, 0.0), (1.0, 1.0)))

    diagnosis = diagnose_persistent_dynamic_blockage(scenario, trace)

    assert diagnosis.blocking_pedestrian_ids == (0, 1)
    assert diagnosis.blocking_cells == ((1, 0), (1, 1))
    assert diagnosis.blocker_count == 2
    assert diagnosis.multiple_humans_jointly_form_cut
    assert (
        diagnosis.persistent_blockage_type
        == "multi_human_corridor_cut_set_blockage"
    )


def test_nearby_terminal_pedestrian_without_cut_is_not_blockage() -> None:
    scenario = _scenario(
        height=2,
        pedestrians=(_terminal_pedestrian((0.0, 1.0)),),
    )
    trace = _trace(((0.0, 1.0),))

    diagnosis = diagnose_persistent_dynamic_blockage(scenario, trace)

    assert diagnosis.static_connectivity
    assert diagnosis.connectivity_with_persistent_blockers
    assert not diagnosis.persistent_dynamic_blockage
    assert diagnosis.blocking_pedestrian_ids == ()
    assert diagnosis.diagnostic_failure_category == "time_horizon_exhausted"
    assert diagnosis.pedestrian_evidence[0].persistent_terminal_candidate


def test_moving_pedestrian_is_not_a_persistent_terminal_blocker() -> None:
    pedestrian = PedestrianSpec(
        start=(1.0, 0.0),
        target=(2.0, 0.0),
        speed=1.0,
    )
    scenario = _scenario(pedestrians=(pedestrian,))
    trace = _trace(((1.0, 0.0),))

    diagnosis = diagnose_persistent_dynamic_blockage(scenario, trace)
    evidence = diagnosis.pedestrian_evidence[0]

    assert not evidence.persistently_collision_relevant
    assert not evidence.terminal
    assert not evidence.stationary
    assert not evidence.persistent_terminal_candidate
    assert not diagnosis.persistent_dynamic_blockage
    assert diagnosis.connectivity_with_persistent_blockers


def test_static_unreachable_case_stays_distinct() -> None:
    scenario = _scenario(obstacles=((1, 0),))
    trace = _trace(
        (),
        planner_failure_reason="goal_unreachable_static",
    )

    diagnosis = diagnose_persistent_dynamic_blockage(scenario, trace)

    assert not diagnosis.static_connectivity
    assert not diagnosis.persistent_dynamic_blockage
    assert diagnosis.blocking_pedestrian_ids == ()
    assert diagnosis.diagnostic_failure_category == "static_goal_unreachable"
    assert diagnosis.planner_failure_reason == "goal_unreachable_static"


def test_horizon_exhaustion_without_persistent_cut_is_not_misclassified() -> None:
    scenario = _scenario()
    trace = _trace(())

    diagnosis = diagnose_persistent_dynamic_blockage(scenario, trace)

    assert diagnosis.static_connectivity
    assert diagnosis.connectivity_with_persistent_blockers
    assert not diagnosis.persistent_dynamic_blockage
    assert diagnosis.diagnostic_failure_category == "time_horizon_exhausted"
    assert diagnosis.planner_failure_reason == "time_horizon_exhausted"


def test_raw_reason_and_legacy_classifier_remain_unchanged_by_refinement() -> None:
    scenario = _scenario(
        pedestrians=(_terminal_pedestrian((1.0, 0.0)),),
    )
    trace = _trace(((1.0, 0.0),))
    result = _failure_result()

    diagnosis = diagnose_persistent_dynamic_blockage(scenario, trace)

    assert diagnosis.planner_failure_reason == "time_horizon_exhausted"
    assert diagnosis.diagnostic_failure_category == "persistent_dynamic_blockage"
    assert classify_robust_failure(
        scenario,
        result,
        trace,
    ) == "no_safe_initial_bridge"


def test_robust_reporting_exposes_separate_raw_and_refined_evidence() -> None:
    scenario = _scenario(
        pedestrians=(_terminal_pedestrian((1.0, 0.0)),),
    )
    trace = _trace(((1.0, 0.0),))

    record = summarize_robust_execution(
        [scenario],
        [_failure_result()],
        [trace],
    )["remaining_failures"][0]

    assert record["reason"] == "post_override_replan_failure"
    assert record["planner_failure_reason"] == "time_horizon_exhausted"
    assert record["diagnostic_failure_category"] == "persistent_dynamic_blockage"
    assert record["blocking_pedestrian_ids"] == [0]
    assert record["blocking_cells"] == [[1, 0]]
    assert record["blocker_count"] == 1
    assert record["multiple_humans_jointly_form_cut"] is False
    assert record["blockers"][0]["terminal"] is True
    assert record["blockers"][0]["stationary"] is True
