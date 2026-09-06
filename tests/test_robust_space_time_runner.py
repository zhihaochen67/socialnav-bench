import pytest

from socialnav.benchmark import (
    SUPPORTED_METHODS,
    Scenario,
    generate_diverse_scenarios,
    run_episode_with_trace,
)
from socialnav.benchmark import robust_space_time_runner as robust_runner
from socialnav.env.world import HUMAN_COLLISION_DISTANCE
from socialnav.planners.robust_space_time_planner import (
    RobustSpaceTimePlanningResult,
)


OLD_METHODS = (
    "astar",
    "dynamic",
    "social",
    "social_replan",
    "social_replan_escape",
    "social_replan_recovery",
    "social_predictive",
    "social_predictive_replan",
    "social_spacetime",
    "social_spacetime_replan",
)


def _scenario(
    *,
    scenario_id: str = "robust-runner-test",
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] = (1, 0),
    pedestrian_start: tuple[float, float] = (10.0, 10.0),
    pedestrian_target: tuple[float, float] = (10.0, 10.0),
    pedestrian_speed: float = 0.0,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        grid_width=3,
        grid_height=1,
        obstacle_cells=(),
        start=start,
        goal=goal,
        grid_scale=1.0,
        pedestrian_start=pedestrian_start,
        pedestrian_target=pedestrian_target,
        pedestrian_speed=pedestrian_speed,
    )


def test_all_ten_old_methods_remain_in_their_original_order() -> None:
    assert SUPPORTED_METHODS[:-2] == OLD_METHODS
    assert SUPPORTED_METHODS[-2] == "social_spacetime_robust"
    assert SUPPORTED_METHODS[-1] == "social_spacetime_shielded"


def test_robust_method_is_accepted_and_deterministic() -> None:
    scenario = generate_diverse_scenarios(1, seed=42)[0]

    first = run_episode_with_trace(
        scenario,
        "social_spacetime_robust",
        max_steps=300,
    )
    second = run_episode_with_trace(
        scenario,
        "social_spacetime_robust",
        max_steps=300,
    )

    assert second == first


def test_initial_bridge_trace_is_recorded() -> None:
    _, trace = run_episode_with_trace(
        _scenario(),
        "social_spacetime_robust",
        max_steps=1,
    )

    assert trace.continuous_bridge_attempts == 1
    assert trace.continuous_bridge_successes == 1
    assert trace.continuous_bridge_failures == 0
    assert trace.bridge_target_cells == ((0, 0),)
    assert trace.bridge_distances == (0.0,)
    assert len(trace.bridge_min_predicted_separations) == 1
    assert trace.robust_replan_count == 0


def test_collision_egress_trace_and_directional_execution_are_preserved() -> None:
    result, trace = run_episode_with_trace(
        _scenario(
            start=(1, 0),
            goal=(0, 0),
            pedestrian_start=(1.2, 0.0),
            pedestrian_target=(1.2, 0.0),
        ),
        "social_spacetime_robust",
        max_steps=121,
    )

    assert result.human_collision
    assert result.path_length > 0.0
    assert trace.speed_scales[0] == 0.25
    assert trace.collision_egress_attempts == 1
    assert trace.collision_egress_successes == 1
    assert trace.collision_egress_failures == 0
    assert trace.bridge_target_cells == ((0, 0),)
    assert trace.bridge_distances == (1.0,)
    assert trace.bridge_min_predicted_separations == pytest.approx((0.2,))
    assert (
        trace.bridge_min_predicted_separations[0]
        < HUMAN_COLLISION_DISTANCE
    )
    assert result.path_length < trace.bridge_distances[0]


def test_no_safe_egress_trace_is_recorded() -> None:
    result, trace = run_episode_with_trace(
        _scenario(
            start=(1, 0),
            goal=(2, 0),
            pedestrian_start=(1.0, 0.0),
            pedestrian_target=(1.0, 0.0),
        ),
        "social_spacetime_robust",
        max_steps=1,
    )

    assert result.human_collision
    assert trace.continuous_bridge_attempts == 1
    assert trace.continuous_bridge_failures == 1
    assert trace.collision_egress_attempts == 1
    assert trace.collision_egress_failures == 1
    assert trace.robust_planning_failure_reasons == ("no_safe_egress",)


def test_positive_near_zero_execution_triggers_progress_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        robust_runner,
        "compute_directional_speed_scale",
        lambda *_args, **_kwargs: 1e-15,
    )

    _, trace = run_episode_with_trace(
        _scenario(),
        "social_spacetime_robust",
        max_steps=121,
        replan_stop_steps=2,
    )

    assert trace.progress_stall_events >= 1
    assert trace.exact_zero_stall_events == 0
    assert trace.robust_replan_count >= 1


def test_identical_failed_replan_is_suppressed_in_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_plan = robust_runner._plan
    calls = 0

    def fake_plan(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_plan(*args, **kwargs)
        return RobustSpaceTimePlanningResult(
            plan=None,
            failure_reason="mapping_bridge_failure",
            bridge_candidates=(),
            bridge=None,
            grid_planning_result=None,
        )

    monkeypatch.setattr(robust_runner, "_plan", fake_plan)
    monkeypatch.setattr(
        robust_runner,
        "compute_directional_speed_scale",
        lambda *_args, **_kwargs: 0.0,
    )

    _, trace = run_episode_with_trace(
        _scenario(),
        "social_spacetime_robust",
        max_steps=3,
        replan_stop_steps=1,
    )

    assert calls == 2
    assert trace.exact_zero_stall_events == 3
    assert trace.robust_replan_count == 1
    assert trace.robust_replan_successes == 0
    assert trace.robust_replan_failures == 1
    assert trace.suppressed_duplicate_replans == 1
