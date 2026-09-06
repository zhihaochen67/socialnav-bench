"""Isolated execution loop for robust continuous-start space-time plans."""

from __future__ import annotations

from math import hypot, isfinite

import pybullet as p

from socialnav.env.demo_map import (
    SOCIAL_DISTANCE,
    SOCIAL_WEIGHT,
    grid_to_world,
)
from socialnav.env.pedestrian import Pedestrian
from socialnav.env.world import (
    GOAL_TOLERANCE,
    HUMAN_COLLISION_DISTANCE,
    OBSTACLE_HALF_EXTENT,
    PEDESTRIAN_HEIGHT,
    PEDESTRIAN_RADIUS,
    ROBOT_HEIGHT,
    ROBOT_RADIUS,
    ROBOT_SPEED,
    SIMULATION_STEP,
    SLOW_DISTANCE,
    STOP_DISTANCE,
)
from socialnav.evaluation import EpisodeResult, evaluate_episode
from socialnav.metrics import (
    Position,
    colliding_human_indices,
    compute_path_length,
    compute_per_human_minimum_distances,
)
from socialnav.planners.astar import astar
from socialnav.planners.directional_avoidance import (
    ESCAPE_SPEED_SCALE,
    compute_directional_speed_scale,
)
from socialnav.planners.local_safety_shield import (
    ShieldCandidateEvaluation,
    ShieldTriggerReason,
    evaluate_local_action,
    evaluate_local_candidates,
    select_safe_local_action,
)
from socialnav.planners.robust_space_time_planner import (
    RobustSpaceTimePlan,
    RobustSpaceTimePlanningResult,
    interpolate_bridge_position,
    robust_space_time_social_astar,
)
from socialnav.planners.space_time_planner import (
    SpaceTimeAction,
    SpaceTimePlanningResult,
    SpaceTimeSearchStatistics,
    duration_to_simulation_steps,
)

from .diagnostics import (
    EpisodeTrace,
    ShieldCollisionAttribution,
    did_episode_time_out,
)
from .replanning import world_to_nearest_free_cell
from .robust_execution import (
    FailedReplanSuppressor,
    ProgressStallDetector,
    StallEvent,
)
from .scenario import Scenario, build_scenario_grid
from .space_time_diagnostics import (
    SpaceTimePlanningCall,
    build_space_time_planning_call,
)
from .space_time_runner import (
    _create_cylinder,
    _create_obstacles,
    _record_position,
)

ROBUST_SPACE_TIME_METHOD = "social_spacetime_robust"
SHIELDED_SPACE_TIME_METHOD = "social_spacetime_shielded"


def _pedestrian_states(
    pedestrians: list[Pedestrian],
) -> tuple[tuple[Position, Position, Position], ...]:
    return tuple(
        (
            pedestrian.position,
            pedestrian.velocity,
            pedestrian.target_position,
        )
        for pedestrian in pedestrians
    )


def _action_for_target(
    mapped_start: tuple[int, int],
    target_cell: tuple[int, int],
    start_position: Position,
    target_position: Position,
) -> SpaceTimeAction:
    delta = (
        target_cell[0] - mapped_start[0],
        target_cell[1] - mapped_start[1],
    )
    by_delta: dict[tuple[int, int], SpaceTimeAction] = {
        (0, -1): "UP",
        (1, 0): "RIGHT",
        (0, 1): "DOWN",
        (-1, 0): "LEFT",
    }
    if delta in by_delta:
        return by_delta[delta]
    motion = (
        target_position[0] - start_position[0],
        target_position[1] - start_position[1],
    )
    if motion == (0.0, 0.0):
        return "WAIT"
    if abs(motion[0]) >= abs(motion[1]):
        return "RIGHT" if motion[0] > 0.0 else "LEFT"
    return "DOWN" if motion[1] > 0.0 else "UP"


def _shield_collision_attributions(
    robot_trajectory: list[Position],
    pedestrian_trajectories: list[list[Position]],
    execution_phases: list[str],
    evaluated_actions: list[str | None],
    predicted_separations: list[float | None],
) -> tuple[ShieldCollisionAttribution, ...]:
    first_step = next(
        (
            step
            for step in range(1, len(robot_trajectory))
            if any(
                hypot(
                    robot_trajectory[step][0] - human[step][0],
                    robot_trajectory[step][1] - human[step][1],
                )
                <= HUMAN_COLLISION_DISTANCE
                for human in pedestrian_trajectories
            )
        ),
        None,
    )
    if first_step is None:
        return ()
    control_index = first_step - 1
    robot_motion = (
        robot_trajectory[first_step][0]
        - robot_trajectory[first_step - 1][0],
        robot_trajectory[first_step][1]
        - robot_trajectory[first_step - 1][1],
    )
    attributions = []
    for index, human in enumerate(pedestrian_trajectories):
        if hypot(
            robot_trajectory[first_step][0] - human[first_step][0],
            robot_trajectory[first_step][1] - human[first_step][1],
        ) > HUMAN_COLLISION_DISTANCE:
            continue
        human_motion = (
            human[first_step][0] - human[first_step - 1][0],
            human[first_step][1] - human[first_step - 1][1],
        )
        to_robot = (
            robot_trajectory[first_step - 1][0]
            - human[first_step - 1][0],
            robot_trajectory[first_step - 1][1]
            - human[first_step - 1][1],
        )
        moved_into_robot = (
            hypot(*robot_motion) <= 1e-12
            and hypot(*human_motion) > 0.0
            and human_motion[0] * to_robot[0]
            + human_motion[1] * to_robot[1]
            > 1e-12
        )
        predicted = predicted_separations[control_index]
        attributions.append(
            ShieldCollisionAttribution(
                pedestrian_index=index,
                execution_phase=execution_phases[control_index],
                shield_evaluated_interval=predicted is not None,
                candidate_action_selected=evaluated_actions[control_index],
                predicted_minimum_separation=predicted,
                actual_first_collision_time=first_step * SIMULATION_STEP,
                pedestrian_moved_into_robot=moved_into_robot,
            )
        )
    return tuple(attributions)


def _closest_pedestrian(
    pedestrians: list[Pedestrian],
    robot_position: Position,
) -> Pedestrian | None:
    """Return the pedestrian nearest the robot, with stable index-order ties."""
    if not pedestrians:
        return None
    return min(
        pedestrians,
        key=lambda pedestrian: hypot(
            robot_position[0] - pedestrian.position[0],
            robot_position[1] - pedestrian.position[1],
        ),
    )


def _plan(
    scenario: Scenario,
    actual_start_position: Position,
    mapped_start: tuple[int, int],
    pedestrians: list[Pedestrian],
    max_time_seconds: float,
) -> RobustSpaceTimePlanningResult:
    closest = _closest_pedestrian(pedestrians, actual_start_position)
    return robust_space_time_social_astar(
        build_scenario_grid(scenario),
        actual_start_position,
        mapped_start,
        scenario.goal,
        closest.position if closest is not None else None,
        closest.velocity if closest is not None else (0.0, 0.0),
        closest.target_position if closest is not None else None,
        SOCIAL_DISTANCE,
        SOCIAL_WEIGHT,
        HUMAN_COLLISION_DISTANCE,
        scenario.grid_scale,
        ROBOT_SPEED,
        max_time_seconds,
        additional_pedestrians=tuple(
            (
                pedestrian.position,
                pedestrian.velocity,
                pedestrian.target_position,
            )
            for pedestrian in pedestrians
            if pedestrian is not closest
        ),
    )


def _fallback_grid_result(
    planning_result: RobustSpaceTimePlanningResult,
) -> SpaceTimePlanningResult:
    if planning_result.grid_planning_result is not None:
        return planning_result.grid_planning_result
    return SpaceTimePlanningResult(
        plan=None,
        failure_reason="other",
        first_action_safety=(),
        statistics=SpaceTimeSearchStatistics(
            expanded_states=0,
            generated_states=0,
            maximum_time_index_reached=0,
            planning_horizon_reached=False,
            open_set_exhausted=False,
            goal_reached=False,
            returned_path_length=None,
            planned_wait_count=0,
        ),
    )


def _planning_call(
    *,
    scenario: Scenario,
    planning_result: RobustSpaceTimePlanningResult,
    call_index: int,
    is_initial_plan: bool,
    simulation_step: int,
    remaining_episode_time: float,
    stopped_streak: int,
    total_reactive_stopped_steps: int,
    previous_successful_plan_step: int | None,
    actual_robot_position: Position,
    mapped_start: tuple[int, int],
    pedestrians: list[Pedestrian],
) -> SpaceTimePlanningCall:
    diagnostic_start = (
        planning_result.bridge.target_cell
        if planning_result.bridge is not None
        else mapped_start
    )
    closest = _closest_pedestrian(pedestrians, actual_robot_position)
    return build_space_time_planning_call(
        scenario_id=scenario.scenario_id,
        call_index=call_index,
        is_initial_plan=is_initial_plan,
        simulation_step=simulation_step,
        simulated_episode_time=simulation_step * SIMULATION_STEP,
        remaining_episode_time=remaining_episode_time,
        stopped_streak=stopped_streak,
        total_reactive_stopped_steps=total_reactive_stopped_steps,
        previous_successful_plan_step=previous_successful_plan_step,
        actual_robot_world_position=actual_robot_position,
        mapped_robot_grid_cell=diagnostic_start,
        pedestrian_position=(
            closest.position if closest is not None else None
        ),
        pedestrian_velocity=(
            closest.velocity if closest is not None else (0.0, 0.0)
        ),
        pedestrian_target=(
            closest.target_position if closest is not None else None
        ),
        grid_scale=scenario.grid_scale,
        move_duration=scenario.grid_scale / ROBOT_SPEED,
        collision_distance=HUMAN_COLLISION_DISTANCE,
        planning_result=_fallback_grid_result(planning_result),
        additional_pedestrians=tuple(
            (
                pedestrian.position,
                pedestrian.velocity,
                pedestrian.target_position,
            )
            for pedestrian in pedestrians
            if pedestrian is not closest
        ),
        pedestrian_count=len(pedestrians),
    )


def _bridge_step_count(plan: RobustSpaceTimePlan) -> int:
    if plan.bridge.duration == 0.0:
        return 0
    return duration_to_simulation_steps(
        plan.bridge.duration,
        SIMULATION_STEP,
    )


def _classify_robust_episode_failure(
    result: EpisodeResult,
    planning_failures: list[str],
    progress_stall_events: int,
    robust_replan_successes: int,
) -> str | None:
    if result.success:
        return None
    for reason in (
        "no_safe_egress",
        "mapping_bridge_failure",
        "ordinary_spacetime_planning_failure",
    ):
        if reason in planning_failures:
            return reason
    if progress_stall_events > 0 and robust_replan_successes > 0:
        return "reactive_execution_block"
    if progress_stall_events > 0:
        return "progress_stall"
    return "other"


def run_robust_space_time_episode_with_trace(
    scenario: Scenario,
    method: str,
    *,
    max_steps: int,
    replan_stop_steps: int,
) -> tuple[EpisodeResult, EpisodeTrace]:
    """Execute robust or shielded robust space-time navigation."""
    if method not in (ROBUST_SPACE_TIME_METHOD, SHIELDED_SPACE_TIME_METHOD):
        raise ValueError(
            "robust space-time method must be one of "
            f"{ROBUST_SPACE_TIME_METHOD}, {SHIELDED_SPACE_TIME_METHOD}"
        )
    shielded = method == SHIELDED_SPACE_TIME_METHOD
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if replan_stop_steps <= 0:
        raise ValueError("replan_stop_steps must be positive")

    grid_map = build_scenario_grid(scenario)
    astar_path = astar(grid_map, scenario.start, scenario.goal)
    if astar_path is None:
        raise ValueError("scenario goal must be reachable by A*")

    start_position = grid_to_world(scenario.start, scenario.grid_scale)
    pedestrians = [
        Pedestrian(spec.start, spec.target, spec.speed)
        for spec in scenario.pedestrians
    ]
    maximum_planning_seconds = max_steps * SIMULATION_STEP

    continuous_bridge_attempts = 0
    continuous_bridge_successes = 0
    continuous_bridge_failures = 0
    bridge_target_cells: list[tuple[int, int]] = []
    bridge_distances: list[float] = []
    bridge_minimum_separations: list[float] = []
    collision_egress_attempts = 0
    collision_egress_successes = 0
    collision_egress_failures = 0
    robust_planning_failure_reasons: list[str] = []

    def account_planning_result(
        planning_result: RobustSpaceTimePlanningResult,
        actual_robot_position: Position,
    ) -> None:
        nonlocal continuous_bridge_attempts
        nonlocal continuous_bridge_successes
        nonlocal continuous_bridge_failures
        nonlocal collision_egress_attempts
        nonlocal collision_egress_successes
        nonlocal collision_egress_failures

        continuous_bridge_attempts += 1
        starts_unsafe = any(
            hypot(
                actual_robot_position[0] - pedestrian.position[0],
                actual_robot_position[1] - pedestrian.position[1],
            )
            <= HUMAN_COLLISION_DISTANCE
            for pedestrian in pedestrians
        )
        if starts_unsafe:
            collision_egress_attempts += 1
        if planning_result.bridge is None:
            continuous_bridge_failures += 1
            if starts_unsafe:
                collision_egress_failures += 1
        else:
            continuous_bridge_successes += 1
            bridge_target_cells.append(planning_result.bridge.target_cell)
            bridge_distances.append(planning_result.bridge.distance)
            bridge_minimum_separations.append(
                planning_result.bridge.minimum_predicted_separation
            )
            if starts_unsafe:
                collision_egress_successes += 1
        if planning_result.failure_reason is not None:
            robust_planning_failure_reasons.append(
                planning_result.failure_reason
            )

    initial_result = _plan(
        scenario,
        start_position,
        scenario.start,
        pedestrians,
        maximum_planning_seconds,
    )
    account_planning_result(initial_result, start_position)
    last_planning_result = initial_result
    current_plan = initial_result.plan
    selected_plan = current_plan
    planning_calls = [
        _planning_call(
            scenario=scenario,
            planning_result=initial_result,
            call_index=0,
            is_initial_plan=True,
            simulation_step=0,
            remaining_episode_time=maximum_planning_seconds,
            stopped_streak=0,
            total_reactive_stopped_steps=0,
            previous_successful_plan_step=None,
            actual_robot_position=start_position,
            mapped_start=scenario.start,
            pedestrians=pedestrians,
        )
    ]
    previous_successful_plan_step = (
        0 if current_plan is not None else None
    )
    spacetime_plan_count = 1
    spacetime_planning_failures = int(current_plan is None)
    planned_wait_actions = (
        0
        if current_plan is None
        else current_plan.grid_plan.planned_wait_actions
    )
    planned_move_actions = (
        0
        if current_plan is None
        else current_plan.grid_plan.planned_move_actions
    )
    action_steps = duration_to_simulation_steps(
        scenario.grid_scale / ROBOT_SPEED,
        SIMULATION_STEP,
    )
    shortest_path_length = compute_path_length(
        [
            grid_to_world(coordinate, scenario.grid_scale)
            for coordinate in astar_path
        ]
    )
    obstacle_positions = [
        grid_to_world(coordinate, scenario.grid_scale)
        for coordinate in scenario.obstacle_cells
    ]

    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("could not connect to PyBullet DIRECT mode")

    try:
        p.setTimeStep(SIMULATION_STEP, physicsClientId=client_id)
        _create_obstacles(scenario, client_id)
        robot_id = _create_cylinder(
            start_position,
            ROBOT_RADIUS,
            ROBOT_HEIGHT,
            client_id,
        )
        pedestrian_ids = [
            _create_cylinder(
                pedestrian.position,
                PEDESTRIAN_RADIUS,
                PEDESTRIAN_HEIGHT,
                client_id,
            )
            for pedestrian in pedestrians
        ]

        robot_trajectory = [_record_position(robot_id, client_id)]
        pedestrian_trajectories = [
            [_record_position(pedestrian_id, client_id)]
            for pedestrian_id in pedestrian_ids
        ]
        speed_scales: list[float] = []
        stall_detector = ProgressStallDetector(
            initial_position=start_position,
            window_steps=replan_stop_steps,
            robot_speed=ROBOT_SPEED,
            simulation_dt=SIMULATION_STEP,
        )
        suppressor = FailedReplanSuppressor()
        pending_stall_event: StallEvent | None = None
        progress_stall_events = 0
        exact_zero_stall_events = 0
        suppressed_duplicate_replans = 0
        replan_steps: list[int] = []
        robust_replan_successes = 0
        robust_replan_failures = 0
        executed_wait_actions = 0
        total_intentional_wait_steps = 0
        reactive_stopped_steps = 0
        bridge_progress = 0.0
        bridge_steps = (
            0 if current_plan is None else _bridge_step_count(current_plan)
        )
        bridge_start_position = start_position
        action_index = 0
        action_progress = 0.0
        action_start_position: Position | None = None
        override: ShieldCandidateEvaluation | None = None
        override_start_position: Position | None = None
        override_progress = 0
        override_steps = 0
        post_override_replan_pending = False
        shield_checks = 0
        shield_activations = 0
        shield_safe_passthroughs = 0
        unsafe_planned_moves = 0
        unsafe_waits = 0
        local_override_actions: list[str] = []
        local_override_target_cells: list[tuple[int, int]] = []
        local_override_unsafe_human_indices: list[tuple[int, ...]] = []
        candidate_actions_evaluated = 0
        candidate_actions_safe = 0
        shield_trigger_reasons: list[ShieldTriggerReason] = []
        shield_predicted_separations: list[float] = []
        post_override_replans = 0
        post_override_replan_successes = 0
        post_override_replan_failures = 0
        no_safe_local_action_events = 0
        execution_phases: list[str] = []
        evaluated_actions: list[str | None] = []
        per_step_predicted_separations: list[float | None] = []
        steps = 0

        def plan_is_complete() -> bool:
            return (
                current_plan is not None
                and override is None
                and not post_override_replan_pending
                and bridge_progress >= bridge_steps
                and action_index >= len(current_plan.grid_plan.actions)
            )

        while steps < max_steps and not plan_is_complete():
            current_robot_position = robot_trajectory[-1]
            closest = _closest_pedestrian(
                pedestrians,
                current_robot_position,
            )

            if pending_stall_event is not None:
                is_post_override_replan = post_override_replan_pending
                mapped_start = world_to_nearest_free_cell(
                    grid_map,
                    current_robot_position,
                    scenario.grid_scale,
                )
                closest_for_suppress = _closest_pedestrian(
                    pedestrians,
                    current_robot_position,
                )
                if (
                    not is_post_override_replan
                    and suppressor.should_suppress(
                    mapped_start=mapped_start,
                    robot_position=current_robot_position,
                    pedestrian_position=(
                        closest_for_suppress.position
                        if closest_for_suppress is not None
                        else (0.0, 0.0)
                    ),
                    pedestrian_velocity=(
                        closest_for_suppress.velocity
                        if closest_for_suppress is not None
                        else (0.0, 0.0)
                    ),
                    pedestrian_target=(
                        closest_for_suppress.target_position
                        if closest_for_suppress is not None
                        else (0.0, 0.0)
                    ),
                    pedestrian_states=tuple(
                        (
                            pedestrian.position,
                            pedestrian.velocity,
                            pedestrian.target_position,
                        )
                        for pedestrian in pedestrians
                    ),
                    )
                ):
                    suppressed_duplicate_replans += 1
                else:
                    replan_steps.append(steps)
                    if is_post_override_replan:
                        post_override_replans += 1
                    replanned_result = _plan(
                        scenario,
                        current_robot_position,
                        mapped_start,
                        pedestrians,
                        (max_steps - steps) * SIMULATION_STEP,
                    )
                    last_planning_result = replanned_result
                    account_planning_result(
                        replanned_result,
                        current_robot_position,
                    )
                    spacetime_plan_count += 1
                    planning_calls.append(
                        _planning_call(
                            scenario=scenario,
                            planning_result=replanned_result,
                            call_index=len(planning_calls),
                            is_initial_plan=False,
                            simulation_step=steps,
                            remaining_episode_time=(
                                (max_steps - steps) * SIMULATION_STEP
                            ),
                            stopped_streak=(
                                0
                                if is_post_override_replan
                                else replan_stop_steps
                            ),
                            total_reactive_stopped_steps=(
                                reactive_stopped_steps
                            ),
                            previous_successful_plan_step=(
                                previous_successful_plan_step
                            ),
                            actual_robot_position=current_robot_position,
                            mapped_start=mapped_start,
                            pedestrians=pedestrians,
                        )
                    )
                    if replanned_result.plan is None:
                        robust_replan_failures += 1
                        if is_post_override_replan:
                            post_override_replan_failures += 1
                            current_plan = None
                        spacetime_planning_failures += 1
                        assert replanned_result.failure_reason is not None
                        closest_failed = _closest_pedestrian(
                            pedestrians,
                            current_robot_position,
                        )
                        suppressor.record_failure(
                            mapped_start=mapped_start,
                            robot_position=current_robot_position,
                            pedestrian_position=(
                                closest_failed.position
                                if closest_failed is not None
                                else (0.0, 0.0)
                            ),
                            pedestrian_velocity=(
                                closest_failed.velocity
                                if closest_failed is not None
                                else (0.0, 0.0)
                            ),
                            pedestrian_target=(
                                closest_failed.target_position
                                if closest_failed is not None
                                else (0.0, 0.0)
                            ),
                            failure_reason=(
                                replanned_result.failure_reason
                            ),
                            pedestrian_states=tuple(
                                (
                                    pedestrian.position,
                                    pedestrian.velocity,
                                    pedestrian.target_position,
                                )
                                for pedestrian in pedestrians
                            ),
                        )
                    else:
                        robust_replan_successes += 1
                        if is_post_override_replan:
                            post_override_replan_successes += 1
                        previous_successful_plan_step = steps
                        suppressor.record_success()
                        current_plan = replanned_result.plan
                        selected_plan = replanned_result.plan
                        planned_wait_actions += (
                            current_plan.grid_plan.planned_wait_actions
                        )
                        planned_move_actions += (
                            current_plan.grid_plan.planned_move_actions
                        )
                        bridge_progress = 0.0
                        bridge_steps = _bridge_step_count(current_plan)
                        bridge_start_position = current_robot_position
                        action_index = 0
                        action_progress = 0.0
                        action_start_position = None
                stall_detector.reset(current_robot_position)
                pending_stall_event = None
                post_override_replan_pending = False
                if is_post_override_replan and plan_is_complete():
                    continue
            robot_position = current_robot_position
            intentional_wait = False
            execution_phase = "robust_baseline"
            evaluated_action: str | None = None
            predicted_separation: float | None = None
            shield_handled = False

            if shielded:
                if override is None:
                    mapped_for_shield = world_to_nearest_free_cell(
                        grid_map,
                        current_robot_position,
                        scenario.grid_scale,
                    )
                    pedestrian_states = _pedestrian_states(pedestrians)
                    goal_position = grid_to_world(
                        scenario.goal,
                        scenario.grid_scale,
                    )
                    planned_action: SpaceTimeAction
                    planned_target_cell = mapped_for_shield
                    planned_target_position = current_robot_position
                    planned_speed_scale = 1.0
                    planned_is_wait = False
                    reactive_stop = False
                    execution_phase = "no_plan_wait"

                    if current_plan is None:
                        planned_action = "WAIT"
                        planned_is_wait = True
                    elif bridge_progress < bridge_steps:
                        planned_target_cell = current_plan.bridge.target_cell
                        planned_target_position = (
                            current_plan.bridge.target_position
                        )
                        planned_action = _action_for_target(
                            mapped_for_shield,
                            planned_target_cell,
                            current_robot_position,
                            planned_target_position,
                        )
                        intended_motion = (
                            planned_target_position[0]
                            - current_robot_position[0],
                            planned_target_position[1]
                            - current_robot_position[1],
                        )
                        if closest is not None:
                            planned_speed_scale = (
                                compute_directional_speed_scale(
                                    current_robot_position,
                                    closest.position,
                                    intended_motion,
                                    STOP_DISTANCE,
                                    SLOW_DISTANCE,
                                    ESCAPE_SPEED_SCALE,
                                    additional_pedestrian_positions=[
                                        pedestrian.position
                                        for pedestrian in pedestrians
                                        if pedestrian is not closest
                                    ],
                                )
                            )
                        reactive_stop = planned_speed_scale == 0.0
                        if reactive_stop:
                            planned_action = "WAIT"
                            planned_target_cell = mapped_for_shield
                            planned_target_position = current_robot_position
                            planned_is_wait = True
                            execution_phase = "reactive_stop"
                        else:
                            execution_phase = "bridge"
                    else:
                        grid_plan = current_plan.grid_plan
                        planned_action = grid_plan.actions[action_index]
                        if planned_action == "WAIT":
                            planned_is_wait = True
                            execution_phase = "planned_wait"
                        else:
                            next_state = grid_plan.timed_states[
                                action_index + 1
                            ]
                            planned_target_cell = (
                                next_state[0],
                                next_state[1],
                            )
                            planned_target_position = grid_to_world(
                                planned_target_cell,
                                scenario.grid_scale,
                            )
                            intended_motion = (
                                planned_target_position[0]
                                - current_robot_position[0],
                                planned_target_position[1]
                                - current_robot_position[1],
                            )
                            if closest is not None:
                                planned_speed_scale = (
                                    compute_directional_speed_scale(
                                        current_robot_position,
                                        closest.position,
                                        intended_motion,
                                        STOP_DISTANCE,
                                        SLOW_DISTANCE,
                                        ESCAPE_SPEED_SCALE,
                                        additional_pedestrian_positions=[
                                            pedestrian.position
                                            for pedestrian in pedestrians
                                            if pedestrian is not closest
                                        ],
                                    )
                                )
                            reactive_stop = planned_speed_scale == 0.0
                            if reactive_stop:
                                planned_action = "WAIT"
                                planned_target_cell = mapped_for_shield
                                planned_target_position = (
                                    current_robot_position
                                )
                                planned_is_wait = True
                                execution_phase = "reactive_stop"
                            else:
                                execution_phase = "grid_move"

                    planned_evaluation = evaluate_local_action(
                        grid_map,
                        current_robot_position,
                        mapped_for_shield,
                        planned_action,
                        pedestrian_states,
                        grid_scale=scenario.grid_scale,
                        robot_speed=ROBOT_SPEED,
                        collision_distance=HUMAN_COLLISION_DISTANCE,
                        move_speed_scale=(
                            planned_speed_scale
                            if planned_speed_scale > 0.0
                            else 1.0
                        ),
                        target_cell=planned_target_cell,
                        target_position=planned_target_position,
                    )
                    shield_checks += 1
                    evaluated_action = planned_evaluation.action
                    predicted_separation = (
                        planned_evaluation.minimum_predicted_separation
                    )
                    if isfinite(predicted_separation):
                        shield_predicted_separations.append(
                            predicted_separation
                        )

                    if planned_evaluation.safe:
                        shield_safe_passthroughs += 1
                    else:
                        shield_activations += 1
                        if planned_is_wait:
                            unsafe_waits += 1
                        else:
                            unsafe_planned_moves += 1
                        if planned_evaluation.starts_in_collision:
                            trigger_reason: ShieldTriggerReason = (
                                "already_in_collision"
                            )
                        elif reactive_stop:
                            trigger_reason = (
                                "reactive_stop_with_incoming_human"
                            )
                        elif planned_is_wait:
                            trigger_reason = (
                                "stationary_wait_predicted_unsafe"
                            )
                        elif planned_action != "WAIT":
                            trigger_reason = (
                                "planned_move_predicted_unsafe"
                            )
                        else:
                            trigger_reason = "other"
                        shield_trigger_reasons.append(trigger_reason)

                        candidates = evaluate_local_candidates(
                            grid_map,
                            current_robot_position,
                            mapped_for_shield,
                            pedestrian_states,
                            grid_scale=scenario.grid_scale,
                            robot_speed=ROBOT_SPEED,
                            collision_distance=HUMAN_COLLISION_DISTANCE,
                        )
                        candidate_actions_evaluated += len(candidates)
                        candidate_actions_safe += sum(
                            candidate.safe for candidate in candidates
                        )
                        shield_predicted_separations.extend(
                            candidate.minimum_predicted_separation
                            for candidate in candidates
                            if isfinite(
                                candidate.minimum_predicted_separation
                            )
                        )
                        override = select_safe_local_action(
                            candidates,
                            progress_target=goal_position,
                        )
                        if override is None:
                            no_safe_local_action_events += 1
                            shield_handled = True
                            speed_scale = 0.0
                            execution_phase = "no_safe_local_action"
                            evaluated_action = None
                        else:
                            local_override_actions.append(override.action)
                            local_override_target_cells.append(
                                override.target_cell
                            )
                            local_override_unsafe_human_indices.append(
                                planned_evaluation.unsafe_human_indices
                            )
                            override_start_position = (
                                current_robot_position
                            )
                            override_progress = 0
                            override_steps = duration_to_simulation_steps(
                                override.duration,
                                SIMULATION_STEP,
                            )

                if override is not None:
                    assert override_start_position is not None
                    shield_handled = True
                    override_progress += 1
                    fraction = min(
                        override_progress / override_steps,
                        1.0,
                    )
                    robot_position = (
                        override_start_position[0]
                        + (
                            override.target_position[0]
                            - override_start_position[0]
                        )
                        * fraction,
                        override_start_position[1]
                        + (
                            override.target_position[1]
                            - override_start_position[1]
                        )
                        * fraction,
                    )
                    speed_scale = override.speed_scale
                    intentional_wait = override.action == "WAIT"
                    execution_phase = "local_override"
                    evaluated_action = override.action
                    predicted_separation = (
                        override.minimum_predicted_separation
                    )
                    if override_progress >= override_steps:
                        override = None
                        override_start_position = None
                        post_override_replan_pending = True
                        pending_stall_event = "progress_stall"

            if shield_handled:
                pass
            elif current_plan is None:
                speed_scale = 0.0
            elif bridge_progress < bridge_steps:
                intended_motion = (
                    current_plan.bridge.target_position[0]
                    - current_robot_position[0],
                    current_plan.bridge.target_position[1]
                    - current_robot_position[1],
                )
                if closest is None:
                    speed_scale = 1.0
                else:
                    speed_scale = compute_directional_speed_scale(
                        current_robot_position,
                        closest.position,
                        intended_motion,
                        STOP_DISTANCE,
                        SLOW_DISTANCE,
                        ESCAPE_SPEED_SCALE,
                        additional_pedestrian_positions=[
                            pedestrian.position
                            for pedestrian in pedestrians
                            if pedestrian is not closest
                        ],
                    )
                if speed_scale == 0.0:
                    reactive_stopped_steps += 1
                bridge_progress = min(
                    bridge_progress + speed_scale,
                    float(bridge_steps),
                )
                fraction = bridge_progress / bridge_steps
                bridge_for_execution = current_plan.bridge
                if bridge_start_position != current_plan.bridge.start_position:
                    bridge_for_execution = type(current_plan.bridge)(
                        start_position=bridge_start_position,
                        target_cell=current_plan.bridge.target_cell,
                        target_position=current_plan.bridge.target_position,
                        distance=hypot(
                            current_plan.bridge.target_position[0]
                            - bridge_start_position[0],
                            current_plan.bridge.target_position[1]
                            - bridge_start_position[1],
                        ),
                        duration=current_plan.bridge.duration,
                        minimum_predicted_separation=(
                            current_plan.bridge.minimum_predicted_separation
                        ),
                        collision_egress=current_plan.bridge.collision_egress,
                    )
                robot_position = interpolate_bridge_position(
                    bridge_for_execution,
                    fraction,
                )
            else:
                grid_plan = current_plan.grid_plan
                action = grid_plan.actions[action_index]
                if action_start_position is None:
                    action_start_position = current_robot_position
                if action == "WAIT":
                    speed_scale = 1.0
                    intentional_wait = True
                    total_intentional_wait_steps += 1
                    action_progress += 1.0
                    if action_progress >= action_steps:
                        executed_wait_actions += 1
                        action_index += 1
                        action_progress = 0.0
                        action_start_position = None
                else:
                    next_state = grid_plan.timed_states[action_index + 1]
                    target_position = grid_to_world(
                        (next_state[0], next_state[1]),
                        scenario.grid_scale,
                    )
                    intended_motion = (
                        target_position[0] - current_robot_position[0],
                        target_position[1] - current_robot_position[1],
                    )
                    if closest is None:
                        speed_scale = 1.0
                    else:
                        speed_scale = compute_directional_speed_scale(
                            current_robot_position,
                            closest.position,
                            intended_motion,
                            STOP_DISTANCE,
                            SLOW_DISTANCE,
                            ESCAPE_SPEED_SCALE,
                            additional_pedestrian_positions=[
                                pedestrian.position
                                for pedestrian in pedestrians
                                if pedestrian is not closest
                            ],
                        )
                    if speed_scale == 0.0:
                        reactive_stopped_steps += 1
                    action_progress = min(
                        action_progress + speed_scale,
                        float(action_steps),
                    )
                    assert action_start_position is not None
                    fraction = action_progress / action_steps
                    robot_position = (
                        action_start_position[0]
                        + (
                            target_position[0] - action_start_position[0]
                        )
                        * fraction,
                        action_start_position[1]
                        + (
                            target_position[1] - action_start_position[1]
                        )
                        * fraction,
                    )
                    if action_progress >= action_steps:
                        action_index += 1
                        action_progress = 0.0
                        action_start_position = None

            speed_scales.append(speed_scale)
            if shielded:
                execution_phases.append(execution_phase)
                evaluated_actions.append(evaluated_action)
                per_step_predicted_separations.append(
                    predicted_separation
                )
            p.resetBasePositionAndOrientation(
                robot_id,
                (*robot_position, ROBOT_HEIGHT / 2 + 0.01),
                (0.0, 0.0, 0.0, 1.0),
                physicsClientId=client_id,
            )
            for pedestrian, pedestrian_id in zip(
                pedestrians,
                pedestrian_ids,
            ):
                pedestrian_position = pedestrian.advance(SIMULATION_STEP)
                p.resetBasePositionAndOrientation(
                    pedestrian_id,
                    (
                        *pedestrian_position,
                        PEDESTRIAN_HEIGHT / 2 + 0.01,
                    ),
                    (0.0, 0.0, 0.0, 1.0),
                    physicsClientId=client_id,
                )
            p.stepSimulation(physicsClientId=client_id)
            steps += 1
            recorded_robot_position = _record_position(robot_id, client_id)
            robot_trajectory.append(recorded_robot_position)
            for trajectory, pedestrian_id in zip(
                pedestrian_trajectories,
                pedestrian_ids,
            ):
                trajectory.append(_record_position(pedestrian_id, client_id))
            stall_event = stall_detector.observe(
                recorded_robot_position,
                speed_scale,
                intentional_wait=intentional_wait,
            )
            if stall_event == "exact_zero":
                exact_zero_stall_events += 1
                pending_stall_event = stall_event
            elif stall_event == "progress_stall":
                progress_stall_events += 1
                pending_stall_event = stall_event

        result = evaluate_episode(
            robot_trajectory,
            pedestrian_trajectories,
            goal_position=grid_to_world(
                scenario.goal,
                scenario.grid_scale,
            ),
            goal_tolerance=GOAL_TOLERANCE,
            steps=steps,
            dt=SIMULATION_STEP,
            shortest_path_length=shortest_path_length,
            social_distance=SOCIAL_DISTANCE,
            human_collision_distance=HUMAN_COLLISION_DISTANCE,
            obstacle_positions=obstacle_positions,
            obstacle_half_extent=OBSTACLE_HALF_EXTENT,
            robot_radius=ROBOT_RADIUS,
        )
        path_completed = plan_is_complete()
        if selected_plan is None:
            planned_path = (start_position,)
        else:
            planned_path = (
                selected_plan.bridge.start_position,
                selected_plan.bridge.target_position,
                *(
                    grid_to_world(coordinate, scenario.grid_scale)
                    for coordinate in selected_plan.grid_plan.spatial_path[1:]
                ),
            )
        timed_out = did_episode_time_out(
            steps=steps,
            max_steps=max_steps,
            path_completed=path_completed,
        )
        robust_failure_reason = _classify_robust_episode_failure(
            result,
            robust_planning_failure_reasons,
            progress_stall_events,
            robust_replan_successes,
        )
        if shielded and not result.success:
            if result.human_collision:
                robust_failure_reason = "human_collision"
            elif (
                no_safe_local_action_events
                and sum(
                    hypot(
                        robot_trajectory[-1][0] - pedestrian.position[0],
                        robot_trajectory[-1][1] - pedestrian.position[1],
                    )
                    <= SLOW_DISTANCE
                    for pedestrian in pedestrians
                )
                >= 2
            ):
                robust_failure_reason = "local_multi_human_trap"
            elif no_safe_local_action_events:
                robust_failure_reason = "no_safe_local_action"
            elif post_override_replan_failures:
                robust_failure_reason = "post_override_replan_failure"
            elif shield_activations > 1:
                robust_failure_reason = "repeated_unsafe_state"
            elif timed_out:
                robust_failure_reason = "horizon_timeout"
            elif progress_stall_events or exact_zero_stall_events:
                robust_failure_reason = "execution_deadlock"
            else:
                robust_failure_reason = "other"
        final_robot_position = robot_trajectory[-1]
        closest_final = _closest_pedestrian(
            pedestrians,
            final_robot_position,
        )
        minimum_predicted_separation = min(
            (
                candidate.minimum_predicted_separation
                for candidate in last_planning_result.bridge_candidates
            ),
            default=None,
        )
        trace = EpisodeTrace(
            planned_path=planned_path,
            speed_scales=tuple(speed_scales),
            timed_out=timed_out,
            final_robot_position=final_robot_position,
            final_pedestrian_position=(
                closest_final.position
                if closest_final is not None
                else (0.0, 0.0)
            ),
            max_steps=max_steps,
            replan_count=len(replan_steps),
            replan_steps=tuple(replan_steps),
            successful_replans=robust_replan_successes,
            failed_replans=robust_replan_failures,
            planned_wait_actions=planned_wait_actions,
            executed_wait_actions=executed_wait_actions,
            planned_move_actions=planned_move_actions,
            spacetime_plan_count=spacetime_plan_count,
            spacetime_planning_failures=spacetime_planning_failures,
            total_intentional_wait_steps=total_intentional_wait_steps,
            reactive_stopped_steps=reactive_stopped_steps,
            space_time_planning_calls=tuple(planning_calls),
            continuous_bridge_attempts=continuous_bridge_attempts,
            continuous_bridge_successes=continuous_bridge_successes,
            continuous_bridge_failures=continuous_bridge_failures,
            bridge_target_cells=tuple(bridge_target_cells),
            bridge_distances=tuple(bridge_distances),
            bridge_min_predicted_separations=tuple(
                bridge_minimum_separations
            ),
            collision_egress_attempts=collision_egress_attempts,
            collision_egress_successes=collision_egress_successes,
            collision_egress_failures=collision_egress_failures,
            progress_stall_events=progress_stall_events,
            exact_zero_stall_events=exact_zero_stall_events,
            suppressed_duplicate_replans=suppressed_duplicate_replans,
            robust_replan_count=len(replan_steps),
            robust_replan_successes=robust_replan_successes,
            robust_replan_failures=robust_replan_failures,
            robust_planning_failure_reasons=tuple(
                robust_planning_failure_reasons
            ),
            robust_episode_failure_reason=robust_failure_reason,
            pedestrian_count=len(pedestrians),
            initial_pedestrian_positions=tuple(
                pedestrian.start_position for pedestrian in pedestrians
            ),
            initial_pedestrian_velocities=tuple(
                pedestrian.velocity for pedestrian in pedestrians
            ),
            pedestrian_targets=tuple(
                pedestrian.target_position for pedestrian in pedestrians
            ),
            final_pedestrian_positions=tuple(
                pedestrian.position for pedestrian in pedestrians
            ),
            per_human_minimum_distances=(
                compute_per_human_minimum_distances(
                    robot_trajectory,
                    pedestrian_trajectories,
                )
            ),
            collision_human_indices=colliding_human_indices(
                robot_trajectory,
                pedestrian_trajectories,
                HUMAN_COLLISION_DISTANCE,
            ),
            blocking_human_indices=tuple(
                index
                for index, pedestrian in enumerate(pedestrians)
                if hypot(
                    final_robot_position[0] - pedestrian.position[0],
                    final_robot_position[1] - pedestrian.position[1],
                )
                <= STOP_DISTANCE
            ),
            minimum_predicted_separation=minimum_predicted_separation,
            shield_checks=shield_checks,
            shield_activations=shield_activations,
            shield_safe_passthroughs=shield_safe_passthroughs,
            unsafe_planned_moves=unsafe_planned_moves,
            unsafe_waits=unsafe_waits,
            local_override_count=len(local_override_actions),
            local_override_actions=tuple(local_override_actions),
            local_override_target_cells=tuple(
                local_override_target_cells
            ),
            local_override_unsafe_human_indices=tuple(
                local_override_unsafe_human_indices
            ),
            candidate_actions_evaluated=candidate_actions_evaluated,
            candidate_actions_safe=candidate_actions_safe,
            shield_trigger_reasons=tuple(shield_trigger_reasons),
            shield_min_predicted_separation=min(
                shield_predicted_separations,
                default=None,
            ),
            post_override_replans=post_override_replans,
            post_override_replan_successes=(
                post_override_replan_successes
            ),
            post_override_replan_failures=post_override_replan_failures,
            no_safe_local_action_events=no_safe_local_action_events,
            shield_collision_attributions=(
                _shield_collision_attributions(
                    robot_trajectory,
                    pedestrian_trajectories,
                    execution_phases,
                    evaluated_actions,
                    per_step_predicted_separations,
                )
                if shielded
                else ()
            ),
        )
        return result, trace
    finally:
        p.disconnect(client_id)
