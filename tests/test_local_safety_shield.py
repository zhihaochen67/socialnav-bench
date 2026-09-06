import pytest

from socialnav.env.grid_map import GridMap
from socialnav.planners.local_safety_shield import (
    evaluate_local_action,
    evaluate_local_candidates,
    select_safe_local_action,
)


ROBOT = (2.0, 2.0)
MAPPED = (2, 2)
COLLISION_DISTANCE = 0.34


def _grid() -> GridMap:
    return GridMap(5, 5)


def _evaluate_wait(*pedestrians):
    return evaluate_local_action(
        _grid(),
        ROBOT,
        MAPPED,
        "WAIT",
        pedestrians,
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )


def test_wait_is_safe_without_a_moving_pedestrian() -> None:
    evaluation = _evaluate_wait(((4.0, 4.0), (0.0, 0.0), (4.0, 4.0)))

    assert evaluation.safe
    assert evaluation.duration == pytest.approx(0.5)


def test_wait_is_safe_when_pedestrian_moves_away() -> None:
    evaluation = _evaluate_wait(((2.5, 2.0), (1.0, 0.0), (4.0, 2.0)))

    assert evaluation.safe


def test_wait_is_unsafe_when_pedestrian_moves_toward_robot() -> None:
    evaluation = _evaluate_wait(((2.5, 2.0), (-1.0, 0.0), (1.0, 2.0)))

    assert not evaluation.safe
    assert evaluation.unsafe_human_indices == (0,)


def test_second_pedestrian_can_make_wait_unsafe() -> None:
    evaluation = _evaluate_wait(
        ((4.0, 4.0), (0.0, 0.0), (4.0, 4.0)),
        ((2.0, 2.5), (0.0, -1.0), (2.0, 1.0)),
    )

    assert not evaluation.safe
    assert evaluation.unsafe_human_indices == (1,)


def test_wait_respects_target_clamped_prediction() -> None:
    evaluation = _evaluate_wait(
        ((2.5, 2.0), (-1.0, 0.0), (2.4, 2.0)),
    )

    assert evaluation.safe
    assert evaluation.per_human_separations[0][-1] == pytest.approx(0.4)


def test_safe_planned_move_passes_safety_check() -> None:
    evaluation = evaluate_local_action(
        _grid(),
        ROBOT,
        MAPPED,
        "RIGHT",
        (((4.0, 4.0), (0.0, 0.0), (4.0, 4.0)),),
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )

    assert evaluation.safe
    assert evaluation.target_position == (3.0, 2.0)


def test_planned_move_unsafe_against_pedestrian_is_rejected() -> None:
    evaluation = evaluate_local_action(
        _grid(),
        ROBOT,
        MAPPED,
        "RIGHT",
        (((2.5, 2.0), (0.0, 0.0), (2.5, 2.0)),),
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )

    assert not evaluation.safe
    assert evaluation.unsafe_human_indices == (0,)


def test_candidate_safe_for_a_but_unsafe_for_b_is_rejected() -> None:
    evaluation = evaluate_local_action(
        _grid(),
        ROBOT,
        MAPPED,
        "RIGHT",
        (
            ((4.0, 4.0), (0.0, 0.0), (4.0, 4.0)),
            ((2.5, 2.0), (0.0, 0.0), (2.5, 2.0)),
        ),
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )

    assert not evaluation.safe
    assert evaluation.unsafe_human_indices == (1,)


def test_safe_alternate_move_is_selected_when_move_and_wait_are_unsafe() -> None:
    pedestrians = (
        ((2.5, 2.0), (0.0, 0.0), (2.5, 2.0)),
        ((2.0, 2.5), (0.0, -1.0), (2.0, 1.0)),
    )
    evaluations = evaluate_local_candidates(
        _grid(),
        ROBOT,
        MAPPED,
        pedestrians,
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )

    selected = select_safe_local_action(
        evaluations,
        progress_target=(4.0, 2.0),
    )
    assert selected is not None
    assert selected.action == "UP"


def test_static_obstacle_candidate_is_rejected() -> None:
    grid = _grid()
    grid.add_obstacle((3, 2))

    evaluation = evaluate_local_action(
        grid,
        ROBOT,
        MAPPED,
        "RIGHT",
        (),
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )

    assert not evaluation.safe
    assert evaluation.rejection_reason == "obstacle"


def test_deterministic_action_order_breaks_exact_geometric_tie() -> None:
    evaluations = evaluate_local_candidates(
        _grid(),
        ROBOT,
        MAPPED,
        (),
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )

    selected = select_safe_local_action(
        evaluations,
        progress_target=(3.0, 1.0),
    )
    assert selected is not None
    assert selected.action == "UP"


def test_continuous_actual_robot_pose_sets_move_distance_and_duration() -> None:
    evaluation = evaluate_local_action(
        _grid(),
        (2.2, 2.0),
        MAPPED,
        "RIGHT",
        (),
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )

    assert evaluation.distance == pytest.approx(0.8)
    assert evaluation.duration == pytest.approx(0.4)


def test_already_colliding_robot_can_select_improving_egress() -> None:
    evaluations = evaluate_local_candidates(
        _grid(),
        ROBOT,
        MAPPED,
        (((2.2, 2.0), (0.0, 0.0), (2.2, 2.0)),),
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )

    selected = select_safe_local_action(
        evaluations,
        progress_target=(4.0, 2.0),
    )
    assert selected is not None
    assert selected.action == "LEFT"
    assert selected.collision_egress
    assert selected.speed_scale == pytest.approx(0.25)


def test_escape_from_a_that_worsens_proximity_to_b_is_rejected() -> None:
    evaluation = evaluate_local_action(
        _grid(),
        ROBOT,
        MAPPED,
        "LEFT",
        (
            ((2.2, 2.0), (0.0, 0.0), (2.2, 2.0)),
            ((1.6, 2.0), (0.0, 0.0), (1.6, 2.0)),
        ),
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
        move_speed_scale=0.25,
    )

    assert not evaluation.safe
    assert evaluation.rejection_reason == "non_improving_egress"


def test_no_safe_egress_returns_no_action() -> None:
    pedestrians = tuple(
        ((2.0 + dx, 2.0 + dy), (0.0, 0.0), (2.0 + dx, 2.0 + dy))
        for dx, dy in ((0.2, 0.0), (-0.2, 0.0), (0.0, 0.2), (0.0, -0.2))
    )
    evaluations = evaluate_local_candidates(
        _grid(),
        ROBOT,
        MAPPED,
        pedestrians,
        grid_scale=1.0,
        robot_speed=2.0,
        collision_distance=COLLISION_DISTANCE,
    )

    assert select_safe_local_action(
        evaluations,
        progress_target=(4.0, 2.0),
    ) is None
