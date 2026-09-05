from math import hypot

import pytest

from socialnav.planners.pedestrian_prediction import (
    PREDICTION_HORIZONS,
    PREDICTION_TEMPORAL_WEIGHTS,
    compute_predictive_social_cost,
    predict_pedestrian_positions,
)
from socialnav.planners.social_cost import compute_social_cost


def test_fixed_prediction_policy() -> None:
    assert PREDICTION_HORIZONS == (0.5, 1.0, 1.5)
    assert PREDICTION_TEMPORAL_WEIGHTS == (1.0, 0.75, 0.50, 0.25)


def test_predictions_use_world_units_per_second() -> None:
    predictions = predict_pedestrian_positions(
        (1.0, 2.0),
        (2.0, -1.0),
    )

    assert predictions == ((2.0, 1.5), (3.0, 1.0), (4.0, 0.5))


def test_predictions_clamp_at_known_target() -> None:
    predictions = predict_pedestrian_positions(
        (0.0, 0.0),
        (1.0, 0.0),
        target=(0.75, 0.0),
    )

    assert predictions == ((0.5, 0.0), (0.75, 0.0), (0.75, 0.0))


def test_zero_velocity_predicts_stationary_pedestrian() -> None:
    predictions = predict_pedestrian_positions(
        (2.0, 3.0),
        (0.0, 0.0),
    )

    assert predictions == ((2.0, 3.0),) * 3


def test_prediction_is_deterministic() -> None:
    arguments = ((0.25, 0.75), (0.5, -0.25), PREDICTION_HORIZONS)

    predictions = [
        predict_pedestrian_positions(*arguments, target=(2.0, -0.125))
        for _ in range(10)
    ]

    assert predictions == [predictions[0]] * 10


def test_negative_prediction_horizon_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="prediction horizons must be non-negative",
    ):
        predict_pedestrian_positions((0.0, 0.0), (1.0, 0.0), (-0.1,))


def test_future_position_affects_predictive_cost() -> None:
    without_future = compute_predictive_social_cost(
        (0.0, 0.0),
        (10.0, 10.0),
        (),
        2.0,
        2.0,
    )
    with_future = compute_predictive_social_cost(
        (0.0, 0.0),
        (10.0, 10.0),
        ((0.5, 0.0),),
        2.0,
        2.0,
    )

    assert without_future == 0.0
    assert with_future > without_future


def test_closer_future_position_increases_predictive_cost() -> None:
    farther = compute_predictive_social_cost(
        (0.0, 0.0),
        (10.0, 10.0),
        ((1.5, 0.0),),
        2.0,
        2.0,
        (1.0, 0.75),
    )
    closer = compute_predictive_social_cost(
        (0.0, 0.0),
        (10.0, 10.0),
        ((0.5, 0.0),),
        2.0,
        2.0,
        (1.0, 0.75),
    )

    assert closer > farther


def test_temporal_weights_are_applied_to_each_position() -> None:
    robot = (0.0, 0.0)
    positions = ((0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0))
    expected = sum(
        temporal_weight
        * compute_social_cost(robot, position, 2.0, 3.0)
        for position, temporal_weight in zip(
            positions,
            PREDICTION_TEMPORAL_WEIGHTS,
        )
    )

    actual = compute_predictive_social_cost(
        robot,
        positions[0],
        positions[1:],
        2.0,
        3.0,
    )

    assert actual == pytest.approx(expected)


def test_zero_velocity_represents_persistent_decayed_occupancy() -> None:
    pedestrian = (0.5, 0.0)
    predictions = predict_pedestrian_positions(
        pedestrian,
        (0.0, 0.0),
    )
    base_cost = compute_social_cost((0.0, 0.0), pedestrian, 2.0, 3.0)

    cost = compute_predictive_social_cost(
        (0.0, 0.0),
        pedestrian,
        predictions,
        2.0,
        3.0,
    )

    assert cost == pytest.approx(
        base_cost * sum(PREDICTION_TEMPORAL_WEIGHTS)
    )
    assert hypot(*predictions[-1]) == pytest.approx(0.5)
