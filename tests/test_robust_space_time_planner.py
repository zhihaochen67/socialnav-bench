from math import hypot

import pytest

from socialnav.env.grid_map import GridMap
from socialnav.evaluation import evaluate_episode
from socialnav.planners.robust_space_time_planner import (
    BridgeCandidateEvaluation,
    build_continuous_start_transitions,
    interpolate_bridge_position,
    is_collision_egress_motion_safe,
    robust_space_time_social_astar,
    select_continuous_start_bridge,
)


def _transitions(
    grid_map: GridMap,
    actual_start: tuple[float, float],
    mapped_start: tuple[int, int],
    *,
    pedestrian_position: tuple[float, float] = (10.0, 10.0),
    pedestrian_velocity: tuple[float, float] = (0.0, 0.0),
    pedestrian_target: tuple[float, float] | None = (10.0, 10.0),
    grid_scale: float = 1.0,
    robot_speed: float = 1.0,
    collision_distance: float = 0.34,
):
    return build_continuous_start_transitions(
        grid_map,
        actual_start,
        mapped_start,
        pedestrian_position=pedestrian_position,
        pedestrian_velocity=pedestrian_velocity,
        pedestrian_target=pedestrian_target,
        grid_scale=grid_scale,
        robot_speed=robot_speed,
        collision_distance=collision_distance,
    )


def _candidate(evaluations, target):
    return next(item for item in evaluations if item.target_cell == target)


def test_safe_continuous_pose_bridges_around_unsafe_mapped_center() -> None:
    evaluations = _transitions(
        GridMap(3, 1),
        (0.6, 0.0),
        (1, 0),
        pedestrian_position=(1.0, 0.0),
        pedestrian_target=(1.0, 0.0),
    )

    assert not _candidate(evaluations, (1, 0)).safe
    bridge = select_continuous_start_bridge(evaluations, (0.6, 0.0))
    assert bridge is not None
    assert bridge.target_cell == (0, 0)
    assert bridge.start_position == (0.6, 0.0)


def test_bridge_interpolation_never_teleports() -> None:
    evaluations = _transitions(GridMap(2, 1), (0.4, 0.0), (0, 0))
    bridge = select_continuous_start_bridge(evaluations, (0.4, 0.0))
    assert bridge is not None

    assert interpolate_bridge_position(bridge, 0.0) == (0.4, 0.0)
    assert interpolate_bridge_position(bridge, 0.5) == (0.2, 0.0)
    assert interpolate_bridge_position(bridge, 1.0) == (0.0, 0.0)


def test_bridge_duration_is_euclidean_distance_over_robot_speed() -> None:
    evaluations = _transitions(
        GridMap(2, 1),
        (0.4, 0.0),
        (0, 0),
        robot_speed=2.0,
    )
    bridge = select_continuous_start_bridge(evaluations, (0.4, 0.0))
    assert bridge is not None

    assert bridge.distance == pytest.approx(0.4)
    assert bridge.duration == pytest.approx(0.2)


def test_bridge_checks_matched_start_midpoint_and_end_times() -> None:
    evaluations = _transitions(
        GridMap(2, 1),
        (0.0, 0.0),
        (1, 0),
        pedestrian_position=(0.5, -0.5),
        pedestrian_velocity=(0.0, 1.0),
        pedestrian_target=None,
        collision_distance=0.1,
    )
    crossing = _candidate(evaluations, (1, 0))

    assert crossing.predicted_separation_start == pytest.approx(2**-0.5)
    assert crossing.predicted_separation_midpoint == pytest.approx(0.0)
    assert crossing.predicted_separation_end == pytest.approx(2**-0.5)
    assert not crossing.safe
    assert crossing.rejection_reason == "predicted_collision"


def test_unsafe_bridge_candidate_is_rejected() -> None:
    evaluations = _transitions(
        GridMap(2, 1),
        (0.0, 0.0),
        (1, 0),
        pedestrian_position=(1.0, 0.0),
        pedestrian_target=(1.0, 0.0),
    )

    assert not _candidate(evaluations, (1, 0)).safe


def test_bridge_selection_uses_frozen_deterministic_tie_breaking() -> None:
    common = dict(
        target_position=(0.0, 0.0),
        static_valid=True,
        distance=1.0,
        duration=1.0,
        predicted_separation_start=1.0,
        predicted_separation_midpoint=1.0,
        predicted_separation_end=1.0,
        minimum_predicted_separation=1.0,
        collision_egress=False,
        safe=True,
        rejection_reason=None,
    )
    high_x = BridgeCandidateEvaluation(target_cell=(2, 1), **common)
    low_x = BridgeCandidateEvaluation(target_cell=(0, 1), **common)

    first = select_continuous_start_bridge(
        (high_x, low_x),
        (1.0, 1.0),
    )
    second = select_continuous_start_bridge(
        (low_x, high_x),
        (1.0, 1.0),
    )

    assert first == second
    assert first is not None
    assert first.target_cell == (0, 1)


def test_static_obstacle_candidate_is_rejected() -> None:
    grid_map = GridMap(2, 1)
    grid_map.add_obstacle((1, 0))

    evaluations = _transitions(grid_map, (0.25, 0.0), (0, 0))
    obstacle = _candidate(evaluations, (1, 0))

    assert not obstacle.static_valid
    assert not obstacle.safe
    assert obstacle.rejection_reason == "static_invalid"


def test_grid_prediction_time_includes_bridge_duration() -> None:
    result = robust_space_time_social_astar(
        GridMap(1, 1),
        (0.25, 0.0),
        (0, 0),
        (0, 0),
        (0.0, 1.0),
        (1.0, 0.0),
        None,
        0.7,
        10.0,
        0.1,
        1.0,
        1.0,
        5.0,
    )

    assert result.plan is not None
    assert result.plan.bridge.duration == pytest.approx(0.25)
    assert result.grid_planning_result is not None
    wait = next(
        detail
        for detail in result.grid_planning_result.first_action_safety
        if detail.action == "WAIT"
    )
    assert wait.predicted_separation_start == pytest.approx(
        hypot(0.25, 1.0)
    )


def test_robot_inside_collision_gets_monotonic_egress_bridge() -> None:
    evaluations = _transitions(
        GridMap(3, 1),
        (0.8, 0.0),
        (1, 0),
        pedestrian_position=(1.0, 0.0),
        pedestrian_target=(1.0, 0.0),
    )

    away = _candidate(evaluations, (0, 0))
    toward = _candidate(evaluations, (2, 0))
    assert away.collision_egress
    assert away.safe
    assert away.predicted_separation_start < away.predicted_separation_midpoint
    assert away.predicted_separation_midpoint < away.predicted_separation_end
    assert not toward.safe


def test_toward_human_egress_is_rejected() -> None:
    assert not is_collision_egress_motion_safe(
        (0.2, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.2, 0.1, 0.0),
        0.34,
    )


def test_tangential_non_improving_egress_is_rejected() -> None:
    assert not is_collision_egress_motion_safe(
        (0.2, 0.0),
        (0.2, 1.0),
        (0.0, 0.0),
        (0.2, hypot(0.2, 0.5), hypot(0.2, 1.0)),
        0.34,
    )


def test_monotonically_increasing_away_egress_is_accepted() -> None:
    assert is_collision_egress_motion_safe(
        (0.2, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (0.2, 0.6, 1.0),
        0.34,
    )


def test_safe_start_does_not_receive_egress_exception() -> None:
    assert not is_collision_egress_motion_safe(
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (0.5, 0.1, 1.0),
        0.34,
    )


def test_coincident_robot_and_human_reports_no_safe_egress() -> None:
    result = robust_space_time_social_astar(
        GridMap(3, 1),
        (1.0, 0.0),
        (1, 0),
        (2, 0),
        (1.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.0),
        0.7,
        10.0,
        0.34,
        1.0,
        1.0,
        5.0,
    )

    assert result.plan is None
    assert result.failure_reason == "no_safe_egress"


def test_collision_metric_is_not_hidden_by_egress() -> None:
    result = evaluate_episode(
        [(0.2, 0.0), (1.0, 0.0)],
        [[(0.0, 0.0), (0.0, 0.0)]],
        goal_position=(1.0, 0.0),
        goal_tolerance=0.05,
        steps=1,
        dt=1.0,
        shortest_path_length=0.8,
        social_distance=0.7,
        human_collision_distance=0.34,
    )

    assert result.success
    assert result.human_collision
