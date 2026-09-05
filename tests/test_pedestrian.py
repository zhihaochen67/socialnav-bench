import pytest

from socialnav.env.pedestrian import Pedestrian


def test_initial_position_is_start_position() -> None:
    pedestrian = Pedestrian((1.0, 2.0), (4.0, 6.0), speed=1.0)

    assert pedestrian.position == (1.0, 2.0)
    assert pedestrian.start_position == (1.0, 2.0)
    assert not pedestrian.has_reached_target


def test_motion_is_deterministic() -> None:
    first = Pedestrian((0.0, 0.0), (3.0, 4.0), speed=1.5)
    second = Pedestrian((0.0, 0.0), (3.0, 4.0), speed=1.5)

    for time_step in (0.1, 0.25, 0.4):
        assert first.advance(time_step) == second.advance(time_step)


def test_moves_correct_distance_for_known_time_step() -> None:
    pedestrian = Pedestrian((0.0, 0.0), (3.0, 4.0), speed=2.0)

    position = pedestrian.advance(1.0)

    assert position == pytest.approx((1.2, 1.6))


def test_reaches_target_without_overshooting() -> None:
    pedestrian = Pedestrian((0.0, 0.0), (1.0, 0.0), speed=2.0)

    assert pedestrian.advance(1.0) == (1.0, 0.0)
    assert pedestrian.has_reached_target
    assert pedestrian.advance(10.0) == (1.0, 0.0)


def test_zero_distance_starts_at_target_and_stays_there() -> None:
    pedestrian = Pedestrian((2.0, 3.0), (2.0, 3.0), speed=1.0)

    assert pedestrian.has_reached_target
    assert pedestrian.advance(1.0) == (2.0, 3.0)


def test_negative_speed_is_rejected() -> None:
    with pytest.raises(ValueError, match="speed must be non-negative"):
        Pedestrian((0.0, 0.0), (1.0, 0.0), speed=-1.0)


def test_negative_time_step_is_rejected() -> None:
    pedestrian = Pedestrian((0.0, 0.0), (1.0, 0.0), speed=1.0)

    with pytest.raises(ValueError, match="time_step must be non-negative"):
        pedestrian.advance(-0.1)


def test_velocity_matches_straight_line_speed() -> None:
    pedestrian = Pedestrian((0.0, 0.0), (3.0, 4.0), speed=2.5)

    assert pedestrian.velocity == pytest.approx((1.5, 2.0))
    pedestrian.advance(0.5)
    assert pedestrian.velocity == pytest.approx((1.5, 2.0))


def test_velocity_is_zero_at_target() -> None:
    pedestrian = Pedestrian((1.0, 2.0), (1.0, 2.0), speed=3.0)

    assert pedestrian.velocity == (0.0, 0.0)


def test_velocity_is_zero_when_speed_is_zero() -> None:
    pedestrian = Pedestrian((1.0, 2.0), (4.0, 6.0), speed=0.0)

    assert pedestrian.velocity == (0.0, 0.0)


def test_velocity_repeated_calls_are_deterministic() -> None:
    pedestrian = Pedestrian((0.25, 0.75), (2.0, 3.0), speed=1.25)

    velocities = [pedestrian.velocity for _ in range(10)]

    assert velocities == [velocities[0]] * 10
