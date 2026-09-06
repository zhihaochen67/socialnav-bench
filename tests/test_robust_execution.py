import pytest

from socialnav.benchmark.robust_execution import (
    FailedReplanSuppressor,
    ProgressStallDetector,
)


def _detector(window_steps: int = 3) -> ProgressStallDetector:
    return ProgressStallDetector(
        initial_position=(0.0, 0.0),
        window_steps=window_steps,
        robot_speed=2.0,
        simulation_dt=0.1,
    )


def test_ordinary_motion_is_not_a_progress_stall() -> None:
    detector = _detector()

    assert detector.observe((0.1, 0.0), 1.0) is None
    assert detector.observe((0.2, 0.0), 1.0) is None
    assert detector.observe((0.3, 0.0), 1.0) is None


def test_exact_zero_for_full_window_reports_exact_zero_stall() -> None:
    detector = _detector()

    assert detector.observe((0.0, 0.0), 0.0) is None
    assert detector.observe((0.0, 0.0), 0.0) is None
    assert detector.observe((0.0, 0.0), 0.0) == "exact_zero"


def test_positive_but_negligible_progress_reports_progress_stall() -> None:
    detector = _detector()

    assert detector.observe((1e-6, 0.0), 1e-15) is None
    assert detector.observe((2e-6, 0.0), 1e-15) is None
    assert (
        detector.observe((3e-6, 0.0), 1e-15)
        == "progress_stall"
    )


def test_meaningful_progress_is_not_a_stall() -> None:
    detector = _detector()

    detector.observe((0.1, 0.0), 0.5)
    detector.observe((0.2, 0.0), 0.5)
    assert detector.observe((0.3, 0.0), 0.5) is None


def test_exact_threshold_is_not_a_stall() -> None:
    detector = _detector()

    detector.observe((0.05, 0.0), 0.5)
    detector.observe((0.1, 0.0), 0.5)
    assert detector.observe((0.2, 0.0), 0.5) is None


def test_intentional_wait_never_counts_toward_stall_window() -> None:
    detector = _detector()

    for _ in range(5):
        assert (
            detector.observe(
                (0.0, 0.0),
                1.0,
                intentional_wait=True,
            )
            is None
        )
    assert detector.observe((0.0, 0.0), 0.0) is None


def test_stall_window_boundary_is_deterministic() -> None:
    detector = _detector(window_steps=2)

    assert detector.observe((0.0, 0.0), 0.0) is None
    assert detector.observe((0.0, 0.0), 0.0) == "exact_zero"
    assert detector.observe((0.0, 0.0), 0.0) is None


def test_progress_threshold_is_robot_speed_times_physics_dt() -> None:
    detector = _detector()

    assert detector.minimum_meaningful_displacement == pytest.approx(0.2)


def _state() -> dict[str, object]:
    return {
        "mapped_start": (1, 2),
        "robot_position": (1.1, 2.2),
        "pedestrian_position": (2.0, 2.0),
        "pedestrian_velocity": (0.0, 0.0),
        "pedestrian_target": (2.0, 2.0),
    }


def _record_failure(suppressor: FailedReplanSuppressor) -> None:
    suppressor.record_failure(
        **_state(),
        failure_reason="mapping_bridge_failure",
    )


def test_identical_failed_state_suppresses_expensive_retry() -> None:
    suppressor = FailedReplanSuppressor()
    _record_failure(suppressor)

    assert suppressor.should_suppress(**_state())


def test_changed_pedestrian_position_enables_retry() -> None:
    suppressor = FailedReplanSuppressor()
    _record_failure(suppressor)
    changed = _state()
    changed["pedestrian_position"] = (2.002, 2.0)

    assert not suppressor.should_suppress(**changed)


def test_changed_robot_position_enables_retry() -> None:
    suppressor = FailedReplanSuppressor()
    _record_failure(suppressor)
    changed = _state()
    changed["robot_position"] = (1.102, 2.2)

    assert not suppressor.should_suppress(**changed)


def test_successful_plan_clears_failure_suppression_state() -> None:
    suppressor = FailedReplanSuppressor()
    _record_failure(suppressor)

    suppressor.record_success()

    assert not suppressor.should_suppress(**_state())
    assert suppressor.last_failed_signature is None


def test_duplicate_suppression_is_deterministic() -> None:
    first = FailedReplanSuppressor()
    second = FailedReplanSuppressor()
    _record_failure(first)
    _record_failure(second)

    assert first.last_failed_signature == second.last_failed_signature
    assert first.should_suppress(**_state())
    assert second.should_suppress(**_state())
