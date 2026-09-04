import pytest

from socialnav.planners.social_cost import compute_social_cost


def test_distance_beyond_threshold_has_zero_cost() -> None:
    cost = compute_social_cost((0.0, 0.0), (2.0, 0.0), 1.0, 3.0)

    assert cost == 0.0


def test_distance_exactly_at_threshold_has_zero_cost() -> None:
    cost = compute_social_cost((0.0, 0.0), (1.0, 0.0), 1.0, 3.0)

    assert cost == 0.0


def test_distance_inside_personal_space_has_positive_cost() -> None:
    cost = compute_social_cost((0.0, 0.0), (0.5, 0.0), 1.0, 2.0)

    assert cost == pytest.approx(0.5)
    assert cost > 0.0


def test_closer_pedestrian_has_larger_cost() -> None:
    closer_cost = compute_social_cost((0.0, 0.0), (0.25, 0.0), 1.0, 1.0)
    farther_cost = compute_social_cost((0.0, 0.0), (0.75, 0.0), 1.0, 1.0)

    assert closer_cost > farther_cost


def test_same_position_has_maximum_distance_based_cost() -> None:
    cost = compute_social_cost((2.0, 3.0), (2.0, 3.0), 2.0, 3.0)

    assert cost == pytest.approx(12.0)


def test_zero_weight_has_zero_cost_inside_personal_space() -> None:
    cost = compute_social_cost((0.0, 0.0), (0.25, 0.0), 1.0, 0.0)

    assert cost == 0.0


def test_weight_scales_cost_linearly() -> None:
    base_cost = compute_social_cost((0.0, 0.0), (0.5, 0.0), 1.0, 2.0)
    scaled_cost = compute_social_cost((0.0, 0.0), (0.5, 0.0), 1.0, 6.0)

    assert scaled_cost == pytest.approx(base_cost * 3.0)


def test_euclidean_distance_is_symmetric() -> None:
    first = compute_social_cost((1.0, 2.0), (4.0, 6.0), 6.0, 2.0)
    second = compute_social_cost((4.0, 6.0), (1.0, 2.0), 6.0, 2.0)

    assert first == pytest.approx(2.0)
    assert second == first


@pytest.mark.parametrize("social_distance", [0.0, -1.0])
def test_nonpositive_social_distance_is_rejected(
    social_distance: float,
) -> None:
    with pytest.raises(ValueError, match="social_distance must be positive"):
        compute_social_cost((0.0, 0.0), (1.0, 0.0), social_distance, 1.0)


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="weight must be non-negative"):
        compute_social_cost((0.0, 0.0), (1.0, 0.0), 2.0, -0.1)


def test_repeated_calls_are_deterministic() -> None:
    results = [
        compute_social_cost((0.0, 0.0), (0.6, 0.8), 2.0, 1.5)
        for _ in range(10)
    ]

    assert results == [results[0]] * 10
