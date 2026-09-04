import pytest

from socialnav.metrics.navigation import (
    compute_path_length,
    compute_spl,
    compute_time_to_goal,
    is_success,
)


def test_empty_trajectory_has_zero_path_length() -> None:
    assert compute_path_length([]) == 0.0


def test_single_point_has_zero_path_length() -> None:
    assert compute_path_length([(2.0, 3.0)]) == 0.0


def test_straight_line_path_length_uses_euclidean_distance() -> None:
    assert compute_path_length([(0.0, 0.0), (3.0, 4.0)]) == 5.0


def test_multisegment_path_length_sums_each_segment() -> None:
    trajectory = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]

    assert compute_path_length(trajectory) == 7.0


def test_path_length_does_not_mutate_input() -> None:
    trajectory = [(0.0, 0.0), (1.0, 1.0)]
    original = trajectory.copy()

    compute_path_length(trajectory)

    assert trajectory == original


def test_success_inside_tolerance() -> None:
    assert is_success((0.3, 0.4), (0.0, 0.0), tolerance=0.6)


def test_success_exactly_at_tolerance() -> None:
    assert is_success((0.3, 0.4), (0.0, 0.0), tolerance=0.5)


def test_failure_outside_tolerance() -> None:
    assert not is_success((0.3, 0.4), (0.0, 0.0), tolerance=0.49)


def test_negative_success_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        is_success((0.0, 0.0), (0.0, 0.0), tolerance=-0.1)


def test_time_to_goal_for_successful_episode() -> None:
    assert compute_time_to_goal(steps=12, dt=0.25, success=True) == 3.0


def test_time_to_goal_for_unsuccessful_episode_is_none() -> None:
    assert compute_time_to_goal(steps=12, dt=0.25, success=False) is None


def test_negative_step_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="steps must be non-negative"):
        compute_time_to_goal(steps=-1, dt=0.1, success=True)


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_nonpositive_time_step_is_rejected(dt: float) -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        compute_time_to_goal(steps=1, dt=dt, success=True)


def test_spl_is_zero_for_failed_episode() -> None:
    assert compute_spl(False, 5.0, 8.0) == 0.0


def test_spl_is_one_for_successful_optimal_path() -> None:
    assert compute_spl(True, 5.0, 5.0) == 1.0


def test_spl_penalizes_successful_longer_path() -> None:
    spl = compute_spl(True, 5.0, 8.0)

    assert spl == pytest.approx(0.625)
    assert 0.0 < spl < 1.0


def test_spl_is_one_for_success_at_zero_distance() -> None:
    assert compute_spl(True, 0.0, 0.0) == 1.0


def test_spl_is_zero_for_detour_from_zero_distance_goal() -> None:
    assert compute_spl(True, 0.0, 2.0) == 0.0


@pytest.mark.parametrize(
    ("shortest_path_length", "actual_path_length", "message"),
    [
        (-1.0, 1.0, "shortest_path_length must be non-negative"),
        (1.0, -1.0, "actual_path_length must be non-negative"),
    ],
)
def test_spl_rejects_negative_lengths(
    shortest_path_length: float,
    actual_path_length: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_spl(True, shortest_path_length, actual_path_length)
