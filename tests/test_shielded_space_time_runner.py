from dataclasses import replace

import pytest

from socialnav.benchmark import Scenario, run_episode_with_trace
from socialnav.benchmark import robust_space_time_runner as shielded_runner
from socialnav.planners.robust_space_time_planner import (
    RobustSpaceTimePlanningResult,
)


def _override_scenario() -> Scenario:
    return Scenario(
        scenario_id="shield-override-test",
        grid_width=3,
        grid_height=1,
        obstacle_cells=(),
        start=(0, 0),
        goal=(2, 0),
        grid_scale=1.0,
        pedestrian_start=(0.0, 10.0),
        pedestrian_target=(1.0, 10.0),
        pedestrian_speed=0.5,
    )


def _force_start_action_unsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    real_evaluate = shielded_runner.evaluate_local_action

    def fake_evaluate(*args, **kwargs):
        evaluation = real_evaluate(*args, **kwargs)
        if args[1] == (0.0, 0.0):
            return replace(
                evaluation,
                safe=False,
                unsafe_human_indices=(0,),
                rejection_reason="predicted_collision",
            )
        return evaluation

    monkeypatch.setattr(
        shielded_runner,
        "evaluate_local_action",
        fake_evaluate,
    )


def test_override_moves_physically_advances_human_and_resumes_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_start_action_unsafe(monkeypatch)

    result, trace = run_episode_with_trace(
        _override_scenario(),
        "social_spacetime_shielded",
        max_steps=400,
    )

    assert result.success
    assert result.path_length == pytest.approx(2.0)
    assert result.steps > 2
    assert trace.local_override_count == 1
    assert trace.local_override_actions == ("RIGHT",)
    assert trace.local_override_target_cells == ((1, 0),)
    assert trace.final_pedestrian_positions[0] != (0.0, 10.0)
    assert trace.post_override_replans == 1
    assert trace.post_override_replan_successes == 1
    assert trace.post_override_replan_failures == 0
    assert trace.shield_safe_passthroughs > 0


def test_failed_post_override_replan_stops_conservatively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_start_action_unsafe(monkeypatch)
    real_plan = shielded_runner._plan
    calls = 0

    def fake_plan(*args, **kwargs):
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

    monkeypatch.setattr(shielded_runner, "_plan", fake_plan)

    result, trace = run_episode_with_trace(
        _override_scenario(),
        "social_spacetime_shielded",
        max_steps=121,
    )

    assert not result.success
    assert trace.final_robot_position == pytest.approx((1.0, 0.0))
    assert trace.post_override_replans == 1
    assert trace.post_override_replan_successes == 0
    assert trace.post_override_replan_failures == 1


def test_shielded_execution_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_start_action_unsafe(monkeypatch)

    first = run_episode_with_trace(
        _override_scenario(),
        "social_spacetime_shielded",
        max_steps=400,
    )
    second = run_episode_with_trace(
        _override_scenario(),
        "social_spacetime_shielded",
        max_steps=400,
    )

    assert second == first


def test_already_recorded_collision_remains_a_collision() -> None:
    scenario = Scenario(
        scenario_id="shield-collision-egress-test",
        grid_width=3,
        grid_height=1,
        obstacle_cells=(),
        start=(1, 0),
        goal=(0, 0),
        grid_scale=1.0,
        pedestrian_start=(1.2, 0.0),
        pedestrian_target=(1.2, 0.0),
        pedestrian_speed=0.0,
    )

    result, trace = run_episode_with_trace(
        scenario,
        "social_spacetime_shielded",
        max_steps=10,
    )

    assert result.human_collision
    assert trace.shield_collision_attributions
    assert trace.shield_collision_attributions[0].pedestrian_index == 0
    assert trace.shield_collision_attributions[0].shield_evaluated_interval
