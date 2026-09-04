import pytest

from socialnav.metrics.social import (
    compute_minimum_human_distance,
    compute_social_violation_rate,
    has_human_collision,
)


def test_minimum_distance_for_one_human() -> None:
    robot = [(0.0, 0.0), (1.0, 0.0)]
    humans = [[(0.0, 2.0), (1.0, 1.0)]]

    assert compute_minimum_human_distance(robot, humans) == 1.0


def test_minimum_distance_across_multiple_humans() -> None:
    robot = [(0.0, 0.0), (1.0, 0.0)]
    humans = [
        [(0.0, 2.0), (1.0, 2.0)],
        [(3.0, 0.0), (1.5, 0.0)],
    ]

    assert compute_minimum_human_distance(robot, humans) == 0.5


def test_minimum_distance_can_occur_at_later_timestep() -> None:
    robot = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    humans = [[(0.0, 3.0), (1.0, 2.0), (2.0, 0.25)]]

    assert compute_minimum_human_distance(robot, humans) == 0.25


def test_minimum_distance_without_humans_is_none() -> None:
    assert compute_minimum_human_distance([(0.0, 0.0)], []) is None


def test_minimum_distance_rejects_incompatible_trajectory_lengths() -> None:
    with pytest.raises(ValueError, match="must have length 2"):
        compute_minimum_human_distance(
            [(0.0, 0.0), (1.0, 0.0)],
            [[(0.0, 1.0)]],
        )


def test_no_social_violations() -> None:
    rate = compute_social_violation_rate(
        [(0.0, 0.0), (1.0, 0.0)],
        [[(0.0, 2.0), (1.0, 2.0)]],
        social_distance=1.0,
    )

    assert rate == 0.0


def test_some_timesteps_have_social_violations() -> None:
    rate = compute_social_violation_rate(
        [(0.0, 0.0)] * 4,
        [[(2.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.25, 0.0)]],
        social_distance=1.0,
    )

    assert rate == 0.5


def test_all_timesteps_have_social_violations() -> None:
    rate = compute_social_violation_rate(
        [(0.0, 0.0), (1.0, 0.0)],
        [[(0.5, 0.0), (1.5, 0.0)]],
        social_distance=1.0,
    )

    assert rate == 1.0


def test_multiple_humans_count_once_per_violating_timestep() -> None:
    rate = compute_social_violation_rate(
        [(0.0, 0.0), (0.0, 0.0)],
        [
            [(0.5, 0.0), (2.0, 0.0)],
            [(0.25, 0.0), (3.0, 0.0)],
        ],
        social_distance=1.0,
    )

    assert rate == 0.5


def test_social_violation_rate_without_humans_is_zero() -> None:
    rate = compute_social_violation_rate(
        [(0.0, 0.0)],
        [],
        social_distance=1.0,
    )

    assert rate == 0.0


def test_exact_social_threshold_is_not_a_violation() -> None:
    rate = compute_social_violation_rate(
        [(0.0, 0.0)],
        [[(0.6, 0.8)]],
        social_distance=1.0,
    )

    assert rate == 0.0


@pytest.mark.parametrize("social_distance", [0.0, -0.1])
def test_nonpositive_social_distance_is_rejected(
    social_distance: float,
) -> None:
    with pytest.raises(ValueError, match="social_distance must be positive"):
        compute_social_violation_rate([], [], social_distance)


def test_social_violation_rejects_incompatible_trajectory_lengths() -> None:
    with pytest.raises(ValueError, match="must have length 1"):
        compute_social_violation_rate(
            [(0.0, 0.0)],
            [[(0.0, 1.0), (0.0, 2.0)]],
            social_distance=1.0,
        )


def test_no_human_collision() -> None:
    assert not has_human_collision(
        [(0.0, 0.0), (1.0, 0.0)],
        [[(0.0, 2.0), (1.0, 2.0)]],
        collision_distance=0.25,
    )


def test_human_collision_at_one_timestep() -> None:
    assert has_human_collision(
        [(0.0, 0.0), (1.0, 0.0)],
        [[(0.0, 2.0), (1.1, 0.0)]],
        collision_distance=0.25,
    )


def test_exact_collision_threshold_counts_as_collision() -> None:
    assert has_human_collision(
        [(0.0, 0.0)],
        [[(0.3, 0.4)]],
        collision_distance=0.5,
    )


def test_collision_detection_checks_multiple_humans() -> None:
    assert has_human_collision(
        [(0.0, 0.0), (1.0, 0.0)],
        [
            [(3.0, 0.0), (3.0, 0.0)],
            [(2.0, 0.0), (1.0, 0.0)],
        ],
        collision_distance=0.0,
    )


def test_collision_without_humans_is_false() -> None:
    assert not has_human_collision(
        [(0.0, 0.0)],
        [],
        collision_distance=0.25,
    )


def test_negative_collision_distance_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="collision_distance must be non-negative",
    ):
        has_human_collision([], [], collision_distance=-0.1)


def test_collision_rejects_incompatible_trajectory_lengths() -> None:
    with pytest.raises(ValueError, match="must have length 2"):
        has_human_collision(
            [(0.0, 0.0), (1.0, 0.0)],
            [[(0.0, 1.0)]],
            collision_distance=0.25,
        )


def test_empty_aligned_trajectories_have_no_social_events() -> None:
    humans = [[]]

    assert compute_minimum_human_distance([], humans) is None
    assert compute_social_violation_rate([], humans, 1.0) == 0.0
    assert not has_human_collision([], humans, 0.0)


def test_social_metrics_are_deterministic() -> None:
    robot = [(0.0, 0.0), (1.0, 0.0)]
    humans = [[(0.0, 1.0), (1.0, 0.5)]]

    results = [
        (
            compute_minimum_human_distance(robot, humans),
            compute_social_violation_rate(robot, humans, 0.75),
            has_human_collision(robot, humans, 0.5),
        )
        for _ in range(5)
    ]

    assert results == [results[0]] * 5
