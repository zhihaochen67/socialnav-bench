import pytest

from socialnav.planners.directional_avoidance import (
    DIRECTION_DOT_TOLERANCE,
    ESCAPE_SPEED_SCALE,
    compute_directional_speed_scale,
)
from socialnav.planners.dynamic_avoidance import compute_speed_scale

STOP_DISTANCE = 0.55
SLOW_DISTANCE = 1.25


def _directional_scale(
    robot_position: tuple[float, float],
    pedestrian_position: tuple[float, float],
    intended_motion: tuple[float, float],
    escape_speed_scale: float = ESCAPE_SPEED_SCALE,
) -> float:
    return compute_directional_speed_scale(
        robot_position,
        pedestrian_position,
        intended_motion,
        STOP_DISTANCE,
        SLOW_DISTANCE,
        escape_speed_scale,
    )


def test_far_pedestrian_preserves_full_speed() -> None:
    assert _directional_scale((0.0, 0.0), (2.0, 0.0), (1.0, 0.0)) == 1.0


def test_slow_zone_preserves_existing_distance_based_scale() -> None:
    robot = (0.0, 0.0)
    pedestrian = (0.9, 0.0)

    scale = _directional_scale(robot, pedestrian, (1.0, 0.0))

    assert scale == compute_speed_scale(
        robot,
        pedestrian,
        STOP_DISTANCE,
        SLOW_DISTANCE,
    )
    assert scale == pytest.approx(0.5)


def test_hard_stop_rejects_motion_toward_pedestrian() -> None:
    assert _directional_scale((0.0, 0.0), (0.5, 0.0), (1.0, 0.0)) == 0.0


def test_hard_stop_allows_fixed_speed_directly_away() -> None:
    assert (
        _directional_scale((0.0, 0.0), (0.5, 0.0), (-1.0, 0.0))
        == ESCAPE_SPEED_SCALE
    )


def test_hard_stop_treats_tangential_motion_conservatively() -> None:
    assert _directional_scale((0.0, 0.0), (0.5, 0.0), (0.0, 1.0)) == 0.0


def test_coincident_robot_and_pedestrian_remain_stopped() -> None:
    assert _directional_scale((0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)) == 0.0


def test_zero_intended_motion_remains_stopped() -> None:
    assert _directional_scale((0.0, 0.0), (0.5, 0.0), (0.0, 0.0)) == 0.0


def test_near_zero_away_dot_requires_clear_negative_margin() -> None:
    within_tolerance = -DIRECTION_DOT_TOLERANCE
    clearly_negative = -3.0 * DIRECTION_DOT_TOLERANCE

    assert (
        _directional_scale(
            (0.0, 0.0),
            (0.5, 0.0),
            (within_tolerance, 0.0),
        )
        == 0.0
    )
    assert (
        _directional_scale(
            (0.0, 0.0),
            (0.5, 0.0),
            (clearly_negative, 0.0),
        )
        == ESCAPE_SPEED_SCALE
    )


def test_directional_controller_is_deterministic() -> None:
    arguments = ((1.0, 1.0), (1.4, 1.0), (-0.1, 0.2))

    first = _directional_scale(*arguments)
    repeated = tuple(_directional_scale(*arguments) for _ in range(10))

    assert repeated == (first,) * 10


@pytest.mark.parametrize("escape_speed_scale", (0.0, -0.1, 1.000001))
def test_escape_speed_scale_must_be_in_unit_interval(
    escape_speed_scale: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"escape_speed_scale must be in \(0, 1\]",
    ):
        _directional_scale(
            (0.0, 0.0),
            (0.5, 0.0),
            (-1.0, 0.0),
            escape_speed_scale,
        )
