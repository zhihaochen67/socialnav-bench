import pytest

from socialnav.planners.dynamic_avoidance import compute_speed_scale


def test_pedestrian_beyond_slow_distance_gives_full_speed() -> None:
    assert compute_speed_scale((0.0, 0.0), (2.0, 0.0), 0.5, 1.5) == 1.0


def test_pedestrian_inside_slow_zone_gives_reduced_positive_speed() -> None:
    scale = compute_speed_scale((0.0, 0.0), (1.0, 0.0), 0.5, 1.5)

    assert scale == pytest.approx(0.5)
    assert 0.0 < scale < 1.0


def test_pedestrian_inside_stop_zone_gives_zero_speed() -> None:
    assert compute_speed_scale((0.0, 0.0), (0.25, 0.0), 0.5, 1.5) == 0.0


def test_exact_thresholds_are_inclusive() -> None:
    assert compute_speed_scale((0.0, 0.0), (0.5, 0.0), 0.5, 1.5) == 0.0
    assert compute_speed_scale((0.0, 0.0), (1.5, 0.0), 0.5, 1.5) == 1.0


def test_euclidean_distance_is_symmetric() -> None:
    first = compute_speed_scale((1.0, 2.0), (4.0, 6.0), 4.0, 6.0)
    second = compute_speed_scale((4.0, 6.0), (1.0, 2.0), 4.0, 6.0)

    assert first == pytest.approx(0.5)
    assert second == first


@pytest.mark.parametrize(
    ("stop_distance", "slow_distance"),
    [(-0.1, 1.0), (1.0, 1.0), (2.0, 1.0)],
)
def test_invalid_thresholds_are_rejected(
    stop_distance: float, slow_distance: float
) -> None:
    with pytest.raises(ValueError):
        compute_speed_scale(
            (0.0, 0.0),
            (1.0, 0.0),
            stop_distance,
            slow_distance,
        )


def test_same_position_gives_zero_speed() -> None:
    assert compute_speed_scale((2.0, 3.0), (2.0, 3.0), 0.5, 1.5) == 0.0


def test_repeated_calls_are_deterministic() -> None:
    results = [
        compute_speed_scale((0.0, 0.0), (0.8, 0.6), 0.5, 1.5)
        for _ in range(10)
    ]

    assert results == [results[0]] * 10
