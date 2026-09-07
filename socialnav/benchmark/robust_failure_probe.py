"""Read-only evidence probe for robust space-time benchmark episodes.

The probe executes the official robust runner unchanged and observes it at
its module boundaries (cylinder creation, position recording, bridge
interpolation, physics stepping).  The runner's control flow, planning, and
parameters are never modified, so the replayed episode is identical to the
official run.
"""

from __future__ import annotations

from math import hypot

from socialnav.benchmark.diagnostics import EpisodeTrace
from socialnav.benchmark.failure_analysis import EpisodeEvidence
from socialnav.benchmark.robust_space_time_runner import (
    ROBUST_SPACE_TIME_METHOD,
    SHIELDED_SPACE_TIME_METHOD,
    run_robust_space_time_episode_with_trace,
)
from socialnav.benchmark.runner import (
    MAX_EPISODE_STEPS,
)
from socialnav.benchmark.scenario import Scenario
from socialnav.evaluation import EpisodeResult
from socialnav.metrics import Position
from socialnav.benchmark.replanning import REPLAN_STOP_STEPS

import socialnav.benchmark.robust_space_time_runner as robust_runner

_PROBE_PATCH_ATTRIBUTES = (
    "_record_position",
    "_create_cylinder",
    "interpolate_bridge_position",
)


def run_robust_episode_with_evidence(
    scenario: Scenario,
    *,
    method: str = ROBUST_SPACE_TIME_METHOD,
    max_steps: int = MAX_EPISODE_STEPS,
    replan_stop_steps: int = REPLAN_STOP_STEPS,
) -> tuple[EpisodeResult, EpisodeTrace, EpisodeEvidence]:
    """Run one robust-family episode and record trajectories plus phases.

    The recorded phases are ``"start"``, ``"no_plan"``, ``"bridge"``,
    ``"WAIT"``, and ``"MOVE"``.  A step is attributed to ``no_plan`` when no
    successful plan was active; otherwise zero-displacement steps with a
    zero speed scale are blocked MOVEs, zero-displacement steps with a
    nonzero speed scale are intentional WAITs, and bridge steps are detected
    directly from bridge interpolation calls.
    """
    if method not in (
        ROBUST_SPACE_TIME_METHOD,
        SHIELDED_SPACE_TIME_METHOD,
    ):
        raise ValueError(
            "evidence probe method must be one of "
            f"{ROBUST_SPACE_TIME_METHOD}, {SHIELDED_SPACE_TIME_METHOD}"
        )

    positions_by_body: dict[int, list[Position]] = {}
    radius_by_body: dict[int, float] = {}
    step_bridge_flags: list[bool] = []
    bridge_flag_state = [False]

    original_record = robust_runner._record_position
    original_create = robust_runner._create_cylinder
    original_interpolate = robust_runner.interpolate_bridge_position
    original_step = robust_runner.p.stepSimulation

    def wrapped_create(
        position: Position,
        radius: float,
        height: float,
        client_id: int,
    ) -> int:
        body_id = original_create(position, radius, height, client_id)
        radius_by_body[body_id] = radius
        positions_by_body[body_id] = []
        return body_id

    def wrapped_record(body_id: int, client_id: int) -> Position:
        position = original_record(body_id, client_id)
        positions_by_body.setdefault(body_id, []).append(position)
        return position

    def wrapped_interpolate(bridge: object, fraction: float) -> Position:
        bridge_flag_state[0] = True
        return original_interpolate(bridge, fraction)

    def wrapped_step(*args: object, **kwargs: object) -> object:
        result = original_step(*args, **kwargs)
        step_bridge_flags.append(bridge_flag_state[0])
        bridge_flag_state[0] = False
        return result

    robust_runner._record_position = wrapped_record
    robust_runner._create_cylinder = wrapped_create
    robust_runner.interpolate_bridge_position = wrapped_interpolate
    robust_runner.p.stepSimulation = wrapped_step
    try:
        result, trace = run_robust_space_time_episode_with_trace(
            scenario,
            method,
            max_steps=max_steps,
            replan_stop_steps=replan_stop_steps,
        )
    finally:
        robust_runner._record_position = original_record
        robust_runner._create_cylinder = original_create
        robust_runner.interpolate_bridge_position = original_interpolate
        robust_runner.p.stepSimulation = original_step

    robot_body = next(
        body_id
        for body_id, positions in positions_by_body.items()
        if positions and positions[-1] == trace.final_robot_position
    )
    robot_trajectory = tuple(positions_by_body[robot_body])
    # radius_by_body preserves creation order: the robot was created first,
    # then the pedestrians in scenario index order.
    human_trajectories = tuple(
        tuple(positions_by_body[body_id])
        for body_id in radius_by_body
        if body_id != robot_body
    )

    speed_scales = trace.speed_scales
    planning_calls = trace.space_time_planning_calls
    steps = result.steps
    phases = ["start"]
    for step_index in range(1, steps + 1):
        if step_bridge_flags[step_index - 1]:
            phase = "bridge"
        else:
            active_call = next(
                (
                    call
                    for call in reversed(planning_calls)
                    if call.simulation_step < step_index
                ),
                None,
            )
            if active_call is not None and active_call.failure_reason is not None:
                phase = "no_plan"
            else:
                displacement = hypot(
                    robot_trajectory[step_index][0]
                    - robot_trajectory[step_index - 1][0],
                    robot_trajectory[step_index][1]
                    - robot_trajectory[step_index - 1][1],
                )
                if displacement == 0.0:
                    phase = (
                        "WAIT"
                        if speed_scales[step_index - 1] != 0.0
                        else "MOVE"
                    )
                else:
                    phase = "MOVE"
        phases.append(phase)

    evidence = EpisodeEvidence(
        robot_trajectory=robot_trajectory,
        human_trajectories=human_trajectories,
        phases=tuple(phases),
    )
    return result, trace, evidence