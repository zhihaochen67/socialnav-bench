from dataclasses import replace

from socialnav.benchmark import Scenario, run_episode_with_trace

from socialnav.benchmark.space_time_diagnostics import (
    build_space_time_planning_call,
    group_repeated_failed_planning_calls,
)
from socialnav.env.grid_map import GridMap
from socialnav.planners.space_time_planner import (
    diagnose_space_time_social_astar,
)


def _diagnose(
    grid_map: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    pedestrian_position: tuple[float, float] = (10.0, 10.0),
    pedestrian_velocity: tuple[float, float] = (0.0, 0.0),
    pedestrian_target: tuple[float, float] = (10.0, 10.0),
    max_time_seconds: float = 10.0,
):
    return diagnose_space_time_social_astar(
        grid_map,
        start,
        goal,
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        0.7,
        10.0,
        0.34,
        1.0,
        1.0,
        max_time_seconds,
    )


def _no_safe_first_action_result():
    return _diagnose(
        GridMap(2, 1),
        (0, 0),
        (1, 0),
        pedestrian_position=(0.4, 0.0),
        pedestrian_velocity=(-0.2, 0.0),
        pedestrian_target=(-1.0, 0.0),
    )


def test_invalid_start_classification() -> None:
    result = _diagnose(GridMap(2, 1), (-1, 0), (1, 0))

    assert result.plan is None
    assert result.failure_reason == "invalid_start"
    assert result.first_action_safety == ()
    assert result.statistics.expanded_states == 0


def test_no_safe_first_action_classification() -> None:
    result = _no_safe_first_action_result()

    assert result.plan is None
    assert result.failure_reason == "no_safe_first_action"
    assert not any(
        detail.rejection_reason is None
        for detail in result.first_action_safety
    )
    assert result.first_action_safety[-1].action == "WAIT"
    assert (
        result.first_action_safety[-1].rejection_reason
        == "predicted_collision"
    )


def test_time_horizon_exhaustion_classification() -> None:
    result = _diagnose(
        GridMap(4, 1),
        (0, 0),
        (3, 0),
        max_time_seconds=2.99,
    )

    assert result.plan is None
    assert result.failure_reason == "time_horizon_exhausted"
    assert result.statistics.maximum_time_index_reached == 2
    assert result.statistics.planning_horizon_reached
    assert result.statistics.open_set_exhausted
    assert not result.statistics.goal_reached


def test_successful_plan_has_no_failure_reason() -> None:
    result = _diagnose(GridMap(3, 1), (0, 0), (2, 0))

    assert result.plan is not None
    assert result.failure_reason is None
    assert result.statistics.goal_reached
    assert not result.statistics.open_set_exhausted
    assert result.statistics.returned_path_length == 2
    assert result.statistics.planned_wait_count == 0


def test_first_action_safety_details_are_complete_and_deterministic() -> None:
    first = _diagnose(GridMap(2, 1), (0, 0), (1, 0))
    second = _diagnose(GridMap(2, 1), (0, 0), (1, 0))

    assert second == first
    details = first.first_action_safety
    assert tuple(detail.action for detail in details) == (
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
    )
    assert details[0].rejection_reason == "outside_map"
    assert details[1].rejection_reason is None
    assert details[2].rejection_reason == "outside_map"
    assert details[3].rejection_reason == "outside_map"
    assert details[4].rejection_reason is None
    assert all(detail.minimum_predicted_separation > 0.0 for detail in details)


def test_actual_vs_mapped_pose_detects_artificial_infeasibility() -> None:
    planning_result = _no_safe_first_action_result()

    call = build_space_time_planning_call(
        scenario_id="mapping-test",
        call_index=1,
        is_initial_plan=False,
        simulation_step=120,
        simulated_episode_time=0.5,
        remaining_episode_time=19.5,
        stopped_streak=120,
        total_reactive_stopped_steps=120,
        previous_successful_plan_step=0,
        actual_robot_world_position=(0.0, -0.75),
        mapped_robot_grid_cell=(0, 0),
        pedestrian_position=(0.4, 0.0),
        pedestrian_velocity=(-0.2, 0.0),
        pedestrian_target=(-1.0, 0.0),
        grid_scale=1.0,
        move_duration=1.0,
        collision_distance=0.34,
        planning_result=planning_result,
    )

    assert call.failure_reason == "no_safe_first_action"
    assert call.mapped_pose_safe_first_actions == ()
    assert call.actual_pose_safe_first_actions == ("RIGHT", "WAIT")
    assert call.mapped_start_artifact
    assert call.actual_to_mapped_center_distance == 0.75
    assert call.actual_robot_pedestrian_distance > (
        call.mapped_center_to_pedestrian_distance
    )


def test_repeated_failure_grouping_is_deterministic() -> None:
    planning_result = _no_safe_first_action_result()
    original = build_space_time_planning_call(
        scenario_id="repeat-test",
        call_index=1,
        is_initial_plan=False,
        simulation_step=120,
        simulated_episode_time=0.5,
        remaining_episode_time=19.5,
        stopped_streak=120,
        total_reactive_stopped_steps=120,
        previous_successful_plan_step=0,
        actual_robot_world_position=(0.0, -0.75),
        mapped_robot_grid_cell=(0, 0),
        pedestrian_position=(0.4, 0.0),
        pedestrian_velocity=(-0.2, 0.0),
        pedestrian_target=(-1.0, 0.0),
        grid_scale=1.0,
        move_duration=1.0,
        collision_distance=0.34,
        planning_result=planning_result,
    )
    near_identical = replace(
        original,
        call_index=2,
        simulation_step=240,
        actual_robot_world_position=(0.0004, -0.7504),
    )
    distinct = replace(
        original,
        call_index=3,
        simulation_step=360,
        actual_robot_world_position=(0.002, -0.75),
    )
    calls = [distinct, near_identical, original]

    first = group_repeated_failed_planning_calls(calls)
    second = group_repeated_failed_planning_calls(calls)

    assert second == first
    assert tuple(group.call_count for group in first) == (2, 1)
    assert first[0].first_simulation_step == 120
    assert first[0].last_simulation_step == 240
    assert first[0].mapped_robot_grid_cell == (0, 0)


def test_runner_records_initial_planning_call_context() -> None:
    scenario = Scenario(
        scenario_id="runner-diagnostic-test",
        grid_width=2,
        grid_height=1,
        obstacle_cells=(),
        start=(0, 0),
        goal=(1, 0),
        grid_scale=0.75,
        pedestrian_start=(10.0, 10.0),
        pedestrian_target=(10.0, 10.0),
        pedestrian_speed=0.0,
    )

    result, trace = run_episode_with_trace(
        scenario,
        "social_spacetime_replan",
        max_steps=1,
    )

    assert result.steps == 1
    assert len(trace.space_time_planning_calls) == 1
    call = trace.space_time_planning_calls[0]
    assert call.is_initial_plan
    assert call.call_index == 0
    assert call.simulation_step == 0
    assert call.simulated_episode_time == 0.0
    assert call.actual_robot_world_position == (0.0, 0.0)
    assert call.mapped_robot_grid_cell == (0, 0)
    assert call.mapped_cell_world_center == (0.0, 0.0)
    assert call.actual_to_mapped_center_distance == 0.0
    assert call.failure_reason == "time_horizon_exhausted"


def test_search_exhaustion_before_horizon_is_classified() -> None:
    result = _diagnose(
        GridMap(2, 1),
        (0, 0),
        (1, 0),
        pedestrian_position=(0.6, 0.0),
        pedestrian_velocity=(-0.2, 0.0),
        pedestrian_target=(-1.0, 0.0),
        max_time_seconds=10.0,
    )

    assert result.plan is None
    assert result.failure_reason == "search_exhausted"
    assert result.first_action_safety[1].rejection_reason == (
        "predicted_collision"
    )
    assert result.first_action_safety[-1].rejection_reason is None
    assert result.statistics.maximum_time_index_reached == 1
    assert not result.statistics.planning_horizon_reached
    assert result.statistics.open_set_exhausted
