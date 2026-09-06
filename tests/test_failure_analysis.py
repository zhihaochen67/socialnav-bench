"""Phase 7C failure-analysis focused tests."""

from math import hypot

import pytest

from socialnav.benchmark import (
    PedestrianSpec,
    Scenario,
    generate_diverse_scenarios,
    run_episode_with_trace,
)
from socialnav.benchmark.diagnostics import EpisodeTrace
from socialnav.benchmark.failure_analysis import (
    EpisodeEvidence,
    analyze_first_collision,
    analyze_local_feasibility,
    classify_planning_vs_execution,
    classify_robust_failure,
    compute_final_state_geometry,
    compute_terminal_blockage_evidence,
    scenario_difficulty_features,
    segment_intersects_route,
    select_representative_cases,
    summarize_search_statistics,
    summarize_trace_counters,
)
from socialnav.benchmark.robust_failure_probe import (
    run_robust_episode_with_evidence,
)
from socialnav.benchmark.space_time_diagnostics import (
    SpaceTimePlanningCall,
)
from socialnav.evaluation import EpisodeResult


def _scenario(pedestrians: tuple[PedestrianSpec, ...]) -> Scenario:
    return Scenario(
        scenario_id="analysis-test",
        grid_width=3,
        grid_height=3,
        obstacle_cells=(),
        start=(0, 0),
        goal=(2, 0),
        grid_scale=1.0,
        pedestrians=pedestrians,
    )


def _result(
    *,
    success: bool = False,
    human_collision: bool = False,
    steps: int = 10,
    minimum_human_distance: float | None = 0.5,
) -> EpisodeResult:
    return EpisodeResult(
        success=success,
        path_length=1.0,
        time_to_goal=1.0 if success else None,
        spl=0.5 if success else 0.0,
        minimum_human_distance=minimum_human_distance,
        social_violation_rate=0.1,
        human_collision=human_collision,
        obstacle_collision=False,
        steps=steps,
    )


def _trace(
    *,
    timed_out: bool = True,
    final_robot_position: tuple[float, float] = (0.5, 0.0),
    final_pedestrian_positions: tuple[tuple[float, float], ...] = (
        (10.0, 10.0),
    ),
    bridge_successes: int = 1,
    bridge_failures: int = 0,
    replan_count: int = 0,
    replan_successes: int = 0,
    replan_failures: int = 0,
    exact_zero_stalls: int = 0,
    progress_stalls: int = 0,
    egress_attempts: int = 0,
    egress_failures: int = 0,
    calls: tuple[SpaceTimePlanningCall, ...] = (),
    speed_scales: tuple[float, ...] = (0.0,),
    planned_path: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
    ),
    pedestrian_count: int = 1,
    blocking_human_indices: tuple[int, ...] = (),
    per_human_minimum_distances: tuple[float | None, ...] | None = None,
    collision_human_indices: tuple[int, ...] = (),
    pedestrian_targets: tuple[tuple[float, float], ...] | None = None,
    initial_pedestrian_positions: tuple[tuple[float, float], ...] | None = None,
    initial_pedestrian_velocities: tuple[tuple[float, float], ...] | None = None,
) -> EpisodeTrace:
    resolved_per_human = (
        tuple(None for _ in final_pedestrian_positions)
        if per_human_minimum_distances is None
        else per_human_minimum_distances
    )
    return EpisodeTrace(
        planned_path=planned_path,
        speed_scales=speed_scales,
        timed_out=timed_out,
        final_robot_position=final_robot_position,
        final_pedestrian_position=final_pedestrian_positions[0],
        max_steps=10,
        robust_replan_count=replan_count,
        robust_replan_successes=replan_successes,
        robust_replan_failures=replan_failures,
        continuous_bridge_successes=bridge_successes,
        continuous_bridge_failures=bridge_failures,
        continuous_bridge_attempts=bridge_successes + bridge_failures,
        exact_zero_stall_events=exact_zero_stalls,
        progress_stall_events=progress_stalls,
        collision_egress_attempts=egress_attempts,
        collision_egress_failures=egress_failures,
        space_time_planning_calls=calls,
        pedestrian_count=pedestrian_count,
        final_pedestrian_positions=final_pedestrian_positions,
        pedestrian_targets=(
            tuple((10.0, 10.0) for _ in final_pedestrian_positions)
            if pedestrian_targets is None
            else pedestrian_targets
        ),
        initial_pedestrian_positions=(
            final_pedestrian_positions
            if initial_pedestrian_positions is None
            else initial_pedestrian_positions
        ),
        initial_pedestrian_velocities=(
            tuple((0.0, 0.0) for _ in final_pedestrian_positions)
            if initial_pedestrian_velocities is None
            else initial_pedestrian_velocities
        ),
        per_human_minimum_distances=resolved_per_human,
        collision_human_indices=collision_human_indices,
        blocking_human_indices=blocking_human_indices,
    )


def _call(failure_reason: str | None) -> SpaceTimePlanningCall:
    return SpaceTimePlanningCall(
        scenario_id="analysis-test",
        call_index=0,
        is_initial_plan=True,
        simulation_step=0,
        simulated_episode_time=0.0,
        remaining_episode_time=20.0,
        stopped_streak=0,
        total_reactive_stopped_steps=0,
        previous_successful_plan_step=None,
        actual_robot_world_position=(0.0, 0.0),
        mapped_robot_grid_cell=(0, 0),
        mapped_cell_world_center=(0.0, 0.0),
        actual_to_mapped_center_distance=0.0,
        pedestrian_position=(10.0, 10.0),
        pedestrian_velocity=(0.0, 0.0),
        pedestrian_target=(10.0, 10.0),
        actual_robot_pedestrian_distance=14.14,
        mapped_center_to_pedestrian_distance=14.14,
        pedestrian_at_target=True,
        mapped_pose_safe_first_actions=(),
        actual_pose_safe_first_actions=(),
        mapped_start_artifact=False,
        failure_reason=failure_reason,
        first_action_safety=(),
        expanded_states=3,
        generated_states=5,
        maximum_time_index_reached=1,
        planning_horizon_reached=False,
        open_set_exhausted=failure_reason is not None,
        goal_reached=failure_reason is None,
        returned_path_length=None,
        planned_wait_count=0,
        returned_actions=(),
        returned_timed_states=(),
    )


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

def test_egress_conflict_category() -> None:
    scenario = _scenario(())
    category = classify_robust_failure(
        scenario,
        _result(human_collision=True),
        _trace(egress_attempts=1, egress_failures=1),
    )

    assert category == "multi_human_egress_conflict"


def test_plain_human_collision_category() -> None:
    scenario = _scenario(())
    category = classify_robust_failure(
        scenario,
        _result(human_collision=True),
        _trace(),
    )

    assert category == "human_collision"


def test_grid_failure_reason_categories() -> None:
    scenario = _scenario(())
    for reason, expected in (
        ("no_safe_first_action", "no_safe_first_action"),
        ("search_exhausted", "spacetime_search_exhausted"),
        ("time_horizon_exhausted", "time_horizon_exhausted"),
        ("goal_unreachable_static", "static_goal_unreachable"),
    ):
        category = classify_robust_failure(
            scenario,
            _result(),
            _trace(calls=(_call(reason),)),
        )
        assert category == expected


def test_repeated_replan_failure_category() -> None:
    scenario = _scenario(())
    category = classify_robust_failure(
        scenario,
        _result(),
        _trace(
            bridge_successes=1,
            replan_count=2,
            replan_failures=2,
        ),
    )

    assert category == "repeated_replan_failure"


def test_no_safe_replan_bridge_category() -> None:
    scenario = _scenario(())
    category = classify_robust_failure(
        scenario,
        _result(),
        _trace(
            bridge_successes=1,
            replan_count=1,
            replan_failures=1,
        ),
    )

    assert category == "no_safe_replan_bridge"


def test_no_safe_initial_bridge_category() -> None:
    scenario = _scenario(())
    category = classify_robust_failure(
        scenario,
        _result(),
        _trace(bridge_successes=0, bridge_failures=1),
    )

    assert category == "no_safe_initial_bridge"


def test_reactive_execution_deadlock_category() -> None:
    scenario = _scenario(())
    category = classify_robust_failure(
        scenario,
        _result(),
        _trace(
            exact_zero_stalls=2,
            replan_successes=1,
            replan_count=1,
        ),
    )

    assert category == "reactive_execution_deadlock"


def test_progress_stall_timeout_category() -> None:
    scenario = _scenario(())
    category = classify_robust_failure(
        scenario,
        _result(),
        _trace(progress_stalls=1),
    )

    assert category == "progress_stall_timeout"


def test_other_category() -> None:
    scenario = _scenario(())
    category = classify_robust_failure(
        scenario,
        _result(),
        _trace(),
    )

    assert category == "other"


def test_taxonomy_is_deterministic() -> None:
    scenario = _scenario(())
    result = _result()
    trace = _trace(calls=(_call("no_safe_first_action"),))

    assert classify_robust_failure(
        scenario, result, trace
    ) == classify_robust_failure(scenario, result, trace)


# ---------------------------------------------------------------------------
# Terminal-blockage detection
# ---------------------------------------------------------------------------

def _terminal_scenario() -> Scenario:
    return Scenario(
        scenario_id="terminal-test",
        grid_width=3,
        grid_height=3,
        obstacle_cells=(),
        start=(0, 0),
        goal=(2, 0),
        grid_scale=1.0,
        pedestrians=(
            PedestrianSpec((0.4, 2.0), (0.4, 0.0), 0.5),
            PedestrianSpec((0.9, 2.0), (0.9, 0.0), 0.5),
        ),
    )


def test_terminal_blockers_identified_near_route() -> None:
    scenario = _terminal_scenario()
    trace = _trace(
        final_robot_position=(0.5, 0.0),
        final_pedestrian_positions=((0.4, 0.0), (0.9, 0.0)),
        pedestrian_count=2,
        planned_path=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
    )

    evidence = compute_terminal_blockage_evidence(scenario, trace)

    assert evidence.terminal_indices == (0, 1)
    assert evidence.terminal_route_blocker_indices == (0, 1)
    assert evidence.joint_corridor_blockage
    assert evidence.robot_near_blocker


def test_corridor_blockage_category() -> None:
    scenario = _terminal_scenario()
    trace = _trace(
        final_robot_position=(0.5, 0.0),
        final_pedestrian_positions=((0.4, 0.0), (0.9, 0.0)),
        pedestrian_count=2,
        planned_path=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
    )

    assert classify_robust_failure(
        scenario, _result(), trace
    ) == "multi_pedestrian_corridor_blockage"


def test_single_terminal_blockage_category() -> None:
    scenario = _scenario(
        (PedestrianSpec((0.4, 2.0), (0.4, 0.0), 0.5),)
    )
    trace = _trace(
        final_robot_position=(0.5, 0.0),
        final_pedestrian_positions=((0.4, 0.0),),
        planned_path=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
    )

    assert classify_robust_failure(
        scenario, _result(), trace
    ) == "pedestrian_terminal_blockage"


# ---------------------------------------------------------------------------
# Planning-vs-execution classes
# ---------------------------------------------------------------------------

def test_planning_vs_execution_classes() -> None:
    scenario = _scenario(())

    assert classify_planning_vs_execution(
        _result(), _trace(bridge_successes=0, bridge_failures=1)
    ) == "planner_no_valid_plan"

    assert classify_planning_vs_execution(
        _result(human_collision=True),
        _trace(egress_attempts=0),
    ) == "execution_progressed_then_collided"

    assert classify_planning_vs_execution(
        _result(),
        _trace(
            replan_count=2,
            replan_failures=2,
        ),
    ) == "repeated_replan_failure_unsafe_state"

    assert classify_planning_vs_execution(
        _result(),
        _trace(exact_zero_stalls=1, replan_successes=1, replan_count=1),
    ) == "planner_ok_execution_deadlock"

    assert classify_planning_vs_execution(
        _result(),
        _trace(),
    ) == "apparently_infeasible_horizon"


# ---------------------------------------------------------------------------
# Collision attribution
# ---------------------------------------------------------------------------

def test_collision_attribution_attributes_first_collision() -> None:
    scenario = _scenario(
        (PedestrianSpec((0.3, 0.0), (0.0, 0.0), 0.5),)
    )
    result = _result(human_collision=True)
    trace = _trace(speed_scales=(0.0,))
    evidence = EpisodeEvidence(
        robot_trajectory=((0.0, 0.0), (0.0, 0.0)),
        human_trajectories=(((0.3, 0.0), (0.2, 0.0)),),
        phases=("start", "MOVE"),
    )

    attribution = analyze_first_collision(
        scenario,
        result,
        trace,
        evidence,
    )

    assert attribution is not None
    assert attribution["first_collision_step"] == 1
    assert attribution["colliding_pedestrian_indices"] == [0]
    assert attribution["robot_stopped"] is True
    assert attribution["pedestrian_moved_into_robot"] is True
    assert attribution["per_human"][0]["human_moved_toward_robot"] is True


def test_collision_attribution_returns_none_without_collision() -> None:
    scenario = _scenario(
        (PedestrianSpec((5.0, 5.0), (5.0, 5.0), 0.0),)
    )
    evidence = EpisodeEvidence(
        robot_trajectory=((0.0, 0.0), (1.0, 0.0)),
        human_trajectories=(((5.0, 5.0), (5.0, 5.0)),),
        phases=("start", "MOVE"),
    )

    assert analyze_first_collision(
        scenario,
        _result(),
        _trace(speed_scales=(1.0,)),
        evidence,
    ) is None


# ---------------------------------------------------------------------------
# Final-state geometry counts
# ---------------------------------------------------------------------------

def test_final_state_geometry_distance_buckets() -> None:
    scenario = _scenario(
        (
            PedestrianSpec((0.3, 0.0), (0.3, 0.0), 0.0),
            PedestrianSpec((0.8, 0.0), (0.8, 0.0), 0.0),
        )
    )
    trace = _trace(
        final_robot_position=(0.0, 0.0),
        final_pedestrian_positions=((0.3, 0.0), (0.8, 0.0)),
        pedestrian_count=2,
        blocking_human_indices=(0,),
        per_human_minimum_distances=(0.3, 0.8),
        collision_human_indices=(),
        initial_pedestrian_positions=((0.3, 0.0), (0.8, 0.0)),
        initial_pedestrian_velocities=((0.0, 0.0), (0.0, 0.0)),
        pedestrian_targets=((0.3, 0.0), (0.8, 0.0)),
        speed_scales=(0.0,),
    )

    geometry = compute_final_state_geometry(
        scenario,
        _result(),
        trace,
    )

    assert geometry["within_collision_distance"] == 1
    assert geometry["within_stop_distance"] == 1
    assert geometry["within_slow_distance"] == 2
    assert geometry["within_social_distance"] == 1
    assert geometry["closest_pedestrian_index"] == 0
    assert geometry["blocking_pedestrian_indices"] == [0]
    assert geometry["per_pedestrian"][0]["terminal_near_failure"] is True


# ---------------------------------------------------------------------------
# Route-intersection feature
# ---------------------------------------------------------------------------

def test_segment_intersects_route_properly() -> None:
    route = ((0.0, 0.0), (1.0, 0.0))

    assert segment_intersects_route((0.5, -1.0), (0.5, 1.0), route)
    assert not segment_intersects_route((0.5, -1.0), (0.5, -0.2), route)
    # Endpoint contact is not a proper crossing.
    assert not segment_intersects_route((0.0, -1.0), (0.0, 1.0), route)


def test_scenario_difficulty_features_are_deterministic() -> None:
    scenario = generate_diverse_scenarios(
        1, seed=42, pedestrian_count=3
    )[0]

    first = scenario_difficulty_features(scenario)
    second = scenario_difficulty_features(scenario)

    assert second == first
    assert first["pedestrian_count"] == 3
    assert first["astar_path_length_moves"] >= 6
    assert first["route_crossing_pedestrians"] >= 0
    assert first["minimum_initial_human_route_distance"] is not None
    assert 0 <= first["approximate_max_simultaneous_nearby_humans"] <= 3


# ---------------------------------------------------------------------------
# Local feasibility
# ---------------------------------------------------------------------------

def test_local_feasibility_reports_action_and_bridge_evidence() -> None:
    scenario = generate_diverse_scenarios(
        1, seed=42, pedestrian_count=3
    )[0]
    result, trace = run_episode_with_trace(
        scenario,
        "social_spacetime_robust",
        max_steps=5,
    )

    feasibility = analyze_local_feasibility(scenario, trace)

    assert len(feasibility["actions"]) == 5
    assert {row["action"] for row in feasibility["actions"]} == {
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
    }
    assert len(feasibility["bridge_candidates"]) == 5
    assert isinstance(feasibility["locally_trapped"], bool)
    assert isinstance(feasibility["multi_human_nearby"], bool)


# ---------------------------------------------------------------------------
# Representative-case selection
# ---------------------------------------------------------------------------

def test_representative_selection_uses_lowest_scenario_id() -> None:
    rows = [
        {
            "density": 3,
            "scenario_id": "diverse-seed-42-episode-0005-pedestrians-3",
            "failure_category": "human_collision",
            "timed_out": False,
            "locally_trapped": False,
        },
        {
            "density": 3,
            "scenario_id": "diverse-seed-42-episode-0001-pedestrians-3",
            "failure_category": "human_collision",
            "timed_out": False,
            "locally_trapped": False,
        },
        {
            "density": 3,
            "scenario_id": "diverse-seed-42-episode-0002-pedestrians-3",
            "failure_category": "progress_stall_timeout",
            "timed_out": True,
            "locally_trapped": False,
        },
        {
            "density": 3,
            "scenario_id": "diverse-seed-42-episode-0003-pedestrians-3",
            "failure_category": "time_horizon_exhausted",
            "timed_out": True,
            "locally_trapped": True,
        },
    ]

    selected = select_representative_cases(
        rows,
        {
            3: [
                {"kind": "collision", "label": "collision"},
                {"kind": "timeout", "label": "timeout"},
                {"kind": "local_trap", "label": "local_trap"},
            ]
        },
    )

    assert selected["n3_collision"]["selected"] == (
        "diverse-seed-42-episode-0001-pedestrians-3"
    )
    assert selected["n3_timeout"]["selected"] == (
        "diverse-seed-42-episode-0002-pedestrians-3"
    )
    assert selected["n3_local_trap"]["selected"] == (
        "diverse-seed-42-episode-0003-pedestrians-3"
    )


def test_representatives_are_distinct_across_selectors() -> None:
    rows = [
        {
            "density": 10,
            "scenario_id": "diverse-seed-42-episode-0000-pedestrians-10",
            "failure_category": "time_horizon_exhausted",
            "timed_out": True,
            "locally_trapped": False,
            "terminal_blockage": {
                "joint_corridor_blockage": False,
                "terminal_route_blocker_indices": [2, 5, 6, 7, 9],
            },
        },
        {
            "density": 10,
            "scenario_id": "diverse-seed-42-episode-0015-pedestrians-10",
            "failure_category": "pedestrian_terminal_blockage",
            "timed_out": True,
            "locally_trapped": False,
            "terminal_blockage": {
                "joint_corridor_blockage": False,
                "terminal_route_blocker_indices": [9],
            },
        },
    ]

    selected = select_representative_cases(
        rows,
        {
            10: [
                {"kind": "terminal_blockage", "label": "terminal_blockage"},
                {
                    "kind": "different_dominant",
                    "label": "different_dominant_failure",
                },
            ]
        },
    )

    assert selected["n10_terminal_blockage"]["selected"] == (
        "diverse-seed-42-episode-0015-pedestrians-10"
    )
    assert selected["n10_different_dominant_failure"]["selected"] == (
        "diverse-seed-42-episode-0000-pedestrians-10"
    )
    assert selected["n10_terminal_blockage"]["selected"] != selected[
        "n10_different_dominant_failure"
    ]["selected"]


def test_terminal_blockage_selector_prefers_dominant_category() -> None:
    rows = [
        {
            "density": 10,
            "scenario_id": "diverse-seed-42-episode-0000-pedestrians-10",
            "failure_category": "time_horizon_exhausted",
            "timed_out": True,
            "locally_trapped": False,
            "terminal_blockage": {
                "joint_corridor_blockage": False,
                "terminal_route_blocker_indices": [2, 5, 6, 7, 9],
            },
        },
        {
            "density": 10,
            "scenario_id": "diverse-seed-42-episode-0015-pedestrians-10",
            "failure_category": "pedestrian_terminal_blockage",
            "timed_out": True,
            "locally_trapped": False,
            "terminal_blockage": {
                "joint_corridor_blockage": False,
                "terminal_route_blocker_indices": [9],
            },
        },
    ]

    selected = select_representative_cases(
        rows,
        {
            10: [
                {"kind": "terminal_blockage", "label": "terminal_blockage"}
            ]
        },
    )

    # The dominant-category episode wins even though episode-0000 has a
    # lower id and more evidence-only blockers.
    assert selected["n10_terminal_blockage"]["selected"] == (
        "diverse-seed-42-episode-0015-pedestrians-10"
    )


def test_corridor_selector_prefers_dominant_category() -> None:
    rows = [
        {
            "density": 5,
            "scenario_id": "diverse-seed-42-episode-0001-pedestrians-5",
            "failure_category": "time_horizon_exhausted",
            "timed_out": True,
            "locally_trapped": False,
            "terminal_blockage": {
                "joint_corridor_blockage": True,
                "terminal_route_blocker_indices": [0, 4],
            },
        },
        {
            "density": 5,
            "scenario_id": "diverse-seed-42-episode-0004-pedestrians-5",
            "failure_category": "multi_pedestrian_corridor_blockage",
            "timed_out": True,
            "locally_trapped": False,
            "terminal_blockage": {
                "joint_corridor_blockage": True,
                "terminal_route_blocker_indices": [0, 1],
            },
        },
    ]

    selected = select_representative_cases(
        rows,
        {
            5: [
                {"kind": "corridor_blockage", "label": "multi_human_blockage"}
            ]
        },
    )

    assert selected["n5_multi_human_blockage"]["selected"] == (
        "diverse-seed-42-episode-0004-pedestrians-5"
    )


def test_representative_selection_reports_missing_cases() -> None:
    rows = [
        {
            "density": 3,
            "scenario_id": "x",
            "failure_category": "human_collision",
            "timed_out": False,
            "locally_trapped": False,
        }
    ]

    selected = select_representative_cases(
        rows,
        {3: [{"kind": "terminal_blockage", "label": "terminal"}]},
    )

    assert selected["n3_terminal"]["selected"] is None


def test_representative_selection_uses_direct_evidence_fields() -> None:
    rows = [
        {
            "density": 5,
            "scenario_id": "diverse-seed-42-episode-0003-pedestrians-5",
            "failure_category": "time_horizon_exhausted",
            "timed_out": True,
            "locally_trapped": False,
            "terminal_blockage": {
                "joint_corridor_blockage": True,
                "terminal_route_blocker_indices": [1, 2],
            },
        },
        {
            "density": 5,
            "scenario_id": "diverse-seed-42-episode-0009-pedestrians-5",
            "failure_category": "human_collision",
            "timed_out": False,
            "locally_trapped": False,
            "terminal_blockage": None,
        },
        {
            "density": 10,
            "scenario_id": "diverse-seed-42-episode-0001-pedestrians-10",
            "failure_category": "time_horizon_exhausted",
            "timed_out": True,
            "locally_trapped": False,
            "terminal_blockage": None,
        },
        {
            "density": 10,
            "scenario_id": "diverse-seed-42-episode-0002-pedestrians-10",
            "failure_category": "human_collision",
            "timed_out": True,
            "locally_trapped": True,
            "terminal_blockage": None,
        },
    ]

    selected = select_representative_cases(
        rows,
        {
            5: [
                {
                    "kind": "corridor_blockage",
                    "label": "multi_human_blockage",
                }
            ],
            10: [
                {
                    "kind": "different_dominant",
                    "label": "different_dominant_failure",
                }
            ],
        },
    )

    assert selected["n5_multi_human_blockage"]["selected"] == (
        "diverse-seed-42-episode-0003-pedestrians-5"
    )
    assert selected["n10_different_dominant_failure"]["selected"] == (
        "diverse-seed-42-episode-0001-pedestrians-10"
    )


# ---------------------------------------------------------------------------
# Deterministic aggregation
# ---------------------------------------------------------------------------

def test_counter_and_search_aggregation_is_deterministic() -> None:
    results = [_result(), _result(success=True)]
    traces = [
        _trace(
            calls=(_call(None), _call("search_exhausted")),
            bridge_successes=1,
            replan_count=1,
            replan_failures=1,
        ),
        _trace(
            calls=(_call(None),),
            bridge_successes=2,
            bridge_failures=1,
        ),
    ]

    first_counters = summarize_trace_counters(results, traces)
    second_counters = summarize_trace_counters(results, traces)
    first_search = summarize_search_statistics(results, traces)
    second_search = summarize_search_statistics(results, traces)

    assert second_counters == first_counters
    assert second_search == first_search
    assert first_search["failures"]["expanded_states"]["mean"] == 3.0
    assert first_search["all"]["planning_calls"]["total"] == 3


# ---------------------------------------------------------------------------
# Evidence probe
# ---------------------------------------------------------------------------

def test_probe_replays_official_runner_identically() -> None:
    scenario = _scenario(
        (PedestrianSpec((10.0, 10.0), (10.0, 10.0), 0.0),)
    )
    official_result, official_trace = run_episode_with_trace(
        scenario,
        "social_spacetime_robust",
        max_steps=2,
    )
    probe_result, probe_trace, evidence = run_robust_episode_with_evidence(
        scenario,
        max_steps=2,
    )

    assert probe_result == official_result
    assert probe_trace == official_trace
    assert len(evidence.robot_trajectory) == official_result.steps + 1
    assert len(evidence.human_trajectories) == 1
    assert len(evidence.phases) == official_result.steps + 1
    assert evidence.phases[0] == "start"
    assert set(evidence.phases) <= {"start", "no_plan", "bridge", "WAIT", "MOVE"}