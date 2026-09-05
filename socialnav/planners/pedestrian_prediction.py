"""Pure constant-velocity pedestrian prediction and social costs."""

from __future__ import annotations

from collections.abc import Iterable
from math import hypot

from socialnav.planners.social_cost import compute_social_cost

Position = tuple[float, float]

PREDICTION_HORIZONS = (0.5, 1.0, 1.5)
PREDICTION_TEMPORAL_WEIGHTS = (1.0, 0.75, 0.50, 0.25)
PREDICTION_EPSILON = 1e-12


def predict_pedestrian_positions(
    position: Position,
    velocity: Position,
    horizons: Iterable[float] = PREDICTION_HORIZONS,
    *,
    target: Position | None = None,
) -> tuple[Position, ...]:
    """Predict positions at fixed horizons, clamping at a known target."""
    prediction_horizons = tuple(horizons)
    if any(horizon < 0.0 for horizon in prediction_horizons):
        raise ValueError("prediction horizons must be non-negative")

    velocity_x, velocity_y = velocity
    speed_squared = velocity_x * velocity_x + velocity_y * velocity_y
    arrival_time: float | None = None
    if target is not None:
        target_x = target[0] - position[0]
        target_y = target[1] - position[1]
        if hypot(target_x, target_y) <= PREDICTION_EPSILON:
            arrival_time = 0.0
        elif speed_squared > 0.0:
            projection_time = (
                target_x * velocity_x + target_y * velocity_y
            ) / speed_squared
            projected_x = position[0] + velocity_x * projection_time
            projected_y = position[1] + velocity_y * projection_time
            target_is_on_velocity_ray = (
                projection_time >= 0.0
                and hypot(
                    projected_x - target[0],
                    projected_y - target[1],
                )
                <= PREDICTION_EPSILON
            )
            if target_is_on_velocity_ray:
                arrival_time = projection_time

    predictions = []
    for horizon in prediction_horizons:
        if (
            target is not None
            and arrival_time is not None
            and horizon + PREDICTION_EPSILON >= arrival_time
        ):
            predictions.append(target)
        else:
            predictions.append(
                (
                    position[0] + velocity_x * horizon,
                    position[1] + velocity_y * horizon,
                )
            )
    return tuple(predictions)


def compute_predictive_social_cost(
    robot_position: Position,
    pedestrian_position: Position,
    predicted_positions: Iterable[Position],
    social_distance: float,
    social_weight: float,
    temporal_weights: Iterable[float] | None = None,
) -> float:
    """Return decayed current-and-future social occupancy cost."""
    positions = (pedestrian_position, *tuple(predicted_positions))
    if temporal_weights is None:
        if len(positions) > len(PREDICTION_TEMPORAL_WEIGHTS):
            raise ValueError("too many positions for default temporal weights")
        weights = PREDICTION_TEMPORAL_WEIGHTS[: len(positions)]
    else:
        weights = tuple(temporal_weights)
    if len(weights) != len(positions):
        raise ValueError("one temporal weight is required per position")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("temporal weights must be non-negative")

    return sum(
        temporal_weight
        * compute_social_cost(
            robot_position,
            occupancy_position,
            social_distance,
            social_weight,
        )
        for occupancy_position, temporal_weight in zip(positions, weights)
    )
