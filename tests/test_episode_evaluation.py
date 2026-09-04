from dataclasses import FrozenInstanceError

import pytest

from socialnav.evaluation import (
    EpisodeResult,
    evaluate_episode,
    has_static_obstacle_collision,
)


def test_evaluates_successful_episode() -> None:
    result = evaluate_episode(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        [[(0.0, 2.0), (1.0, 0.5), (2.0, 2.0)]],
        goal_position=(2.0, 0.0),
        goal_tolerance=0.0,
        steps=2,
        dt=0.5,
        shortest_path_length=2.0,
        social_distance=1.0,
        human_collision_distance=0.25,
    )

    assert result == EpisodeResult(
        success=True,
        path_length=2.0,
        time_to_goal=1.0,
        spl=1.0,
        minimum_human_distance=0.5,
        social_violation_rate=pytest.approx(1.0 / 3.0),
        human_collision=False,
        obstacle_collision=False,
        steps=2,
    )


def test_evaluates_failed_episode() -> None:
    result = evaluate_episode(
        [(0.0, 0.0), (1.0, 0.0)],
        [],
        goal_position=(2.0, 0.0),
        goal_tolerance=0.1,
        steps=1,
        dt=0.25,
        shortest_path_length=2.0,
        social_distance=1.0,
        human_collision_distance=0.25,
    )

    assert not result.success
    assert result.time_to_goal is None
    assert result.spl == 0.0


def test_actual_path_length_is_propagated() -> None:
    result = evaluate_episode(
        [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)],
        [],
        goal_position=(3.0, 4.0),
        goal_tolerance=0.0,
        steps=2,
        dt=0.1,
        shortest_path_length=7.0,
        social_distance=1.0,
        human_collision_distance=0.25,
    )

    assert result.path_length == 7.0


def test_spl_uses_supplied_ordinary_shortest_path_length() -> None:
    result = evaluate_episode(
        [(0.0, 0.0), (0.0, 1.0), (4.0, 1.0), (4.0, 0.0)],
        [],
        goal_position=(4.0, 0.0),
        goal_tolerance=0.0,
        steps=3,
        dt=0.1,
        shortest_path_length=4.0,
        social_distance=1.0,
        human_collision_distance=0.25,
    )

    assert result.path_length == 6.0
    assert result.spl == pytest.approx(4.0 / 6.0)


def test_minimum_human_distance_is_propagated() -> None:
    result = evaluate_episode(
        [(0.0, 0.0), (1.0, 0.0)],
        [[(0.0, 3.0), (1.0, 0.4)]],
        goal_position=(1.0, 0.0),
        goal_tolerance=0.0,
        steps=1,
        dt=0.1,
        shortest_path_length=1.0,
        social_distance=1.0,
        human_collision_distance=0.2,
    )

    assert result.minimum_human_distance == 0.4


def test_social_violation_rate_is_propagated() -> None:
    result = evaluate_episode(
        [(0.0, 0.0)] * 3,
        [[(2.0, 0.0), (0.5, 0.0), (0.25, 0.0)]],
        goal_position=(0.0, 0.0),
        goal_tolerance=0.0,
        steps=2,
        dt=0.1,
        shortest_path_length=0.0,
        social_distance=1.0,
        human_collision_distance=0.1,
    )

    assert result.social_violation_rate == pytest.approx(2.0 / 3.0)


def test_human_collision_is_propagated() -> None:
    result = evaluate_episode(
        [(0.0, 0.0), (1.0, 0.0)],
        [[(0.0, 2.0), (1.25, 0.0)]],
        goal_position=(1.0, 0.0),
        goal_tolerance=0.0,
        steps=1,
        dt=0.1,
        shortest_path_length=1.0,
        social_distance=1.0,
        human_collision_distance=0.25,
    )

    assert result.human_collision


def test_time_to_goal_uses_steps_and_dt() -> None:
    result = evaluate_episode(
        [(0.0, 0.0), (1.0, 0.0)],
        [],
        goal_position=(1.0, 0.0),
        goal_tolerance=0.0,
        steps=1,
        dt=0.125,
        shortest_path_length=1.0,
        social_distance=1.0,
        human_collision_distance=0.25,
    )

    assert result.steps == 1
    assert result.time_to_goal == 0.125


def test_human_trajectories_must_be_timestep_aligned() -> None:
    with pytest.raises(ValueError, match="must have length 2"):
        evaluate_episode(
            [(0.0, 0.0), (1.0, 0.0)],
            [[(0.0, 1.0)]],
            goal_position=(1.0, 0.0),
            goal_tolerance=0.0,
            steps=1,
            dt=0.1,
            shortest_path_length=1.0,
            social_distance=1.0,
            human_collision_distance=0.25,
        )


def test_robot_samples_must_match_step_count() -> None:
    with pytest.raises(ValueError, match="initial state plus one sample"):
        evaluate_episode(
            [(0.0, 0.0), (1.0, 0.0)],
            [],
            goal_position=(1.0, 0.0),
            goal_tolerance=0.0,
            steps=2,
            dt=0.1,
            shortest_path_length=1.0,
            social_distance=1.0,
            human_collision_distance=0.25,
        )


def test_empty_robot_trajectory_is_rejected() -> None:
    with pytest.raises(ValueError, match="must include the initial state"):
        evaluate_episode(
            [],
            [],
            goal_position=(0.0, 0.0),
            goal_tolerance=0.0,
            steps=0,
            dt=0.1,
            shortest_path_length=0.0,
            social_distance=1.0,
            human_collision_distance=0.25,
        )


def test_repeated_episode_evaluation_is_deterministic() -> None:
    arguments = {
        "goal_position": (1.0, 0.0),
        "goal_tolerance": 0.0,
        "steps": 1,
        "dt": 0.1,
        "shortest_path_length": 1.0,
        "social_distance": 1.0,
        "human_collision_distance": 0.25,
    }

    results = [
        evaluate_episode(
            [(0.0, 0.0), (1.0, 0.0)],
            [[(0.0, 2.0), (1.0, 0.5)]],
            **arguments,
        )
        for _ in range(5)
    ]

    assert results == [results[0]] * 5


def test_episode_result_is_immutable() -> None:
    result = EpisodeResult(True, 1.0, 1.0, 1.0, None, 0.0, False, False, 1)

    with pytest.raises(FrozenInstanceError):
        result.success = False  # type: ignore[misc]


def test_static_obstacle_collision_detects_overlap() -> None:
    assert has_static_obstacle_collision(
        [(1.0, 1.0)],
        [(1.0, 1.0)],
        obstacle_half_extent=0.25,
        robot_radius=0.1,
    )


def test_static_obstacle_collision_counts_exact_tangency() -> None:
    assert has_static_obstacle_collision(
        [(0.0, 0.0)],
        [(1.0, 0.0)],
        obstacle_half_extent=0.5,
        robot_radius=0.5,
    )


def test_static_obstacle_collision_returns_false_with_clearance() -> None:
    assert not has_static_obstacle_collision(
        [(0.0, 0.0)],
        [(1.01, 0.0)],
        obstacle_half_extent=0.5,
        robot_radius=0.5,
    )


def test_episode_propagates_static_obstacle_collision() -> None:
    result = evaluate_episode(
        [(0.0, 0.0), (1.0, 0.0)],
        [],
        goal_position=(1.0, 0.0),
        goal_tolerance=0.0,
        steps=1,
        dt=0.1,
        shortest_path_length=1.0,
        social_distance=1.0,
        human_collision_distance=0.25,
        obstacle_positions=[(1.0, 0.0)],
        obstacle_half_extent=0.25,
        robot_radius=0.1,
    )

    assert result.obstacle_collision


@pytest.mark.parametrize(
    ("obstacle_half_extent", "robot_radius", "message"),
    [
        (-0.1, 0.1, "obstacle_half_extent must be non-negative"),
        (0.1, -0.1, "robot_radius must be non-negative"),
    ],
)
def test_static_obstacle_collision_rejects_invalid_geometry(
    obstacle_half_extent: float,
    robot_radius: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        has_static_obstacle_collision(
            [],
            [],
            obstacle_half_extent,
            robot_radius,
        )
