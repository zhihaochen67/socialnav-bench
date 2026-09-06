"""Isolated execution loop for robust continuous-start space-time plans."""

from __future__ import annotations

from math import hypot

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
from socialnav.metrics import Position, compute_path_length
from socialnav.planners.astar import astar
from socialnav.planners.directional_avoidance import (
    ESCAPE_SPEED_SCALE,
    compute_directional_speed_scale,
)
from socialnav.planners.robust_space_time_planner import (
    RobustSpaceTimePlan,
    RobustSpaceTimePlanningResult,
    interpolate_bridge_position,
    robust_space_time_social_astar,
)
from socialnav.planners.space_time_planner import (
    SpaceTimePlanningResult,
    SpaceTimeSearchStatistics,
    duration_to_simulation_steps,
)

from .diagnostics import EpisodeTrace, did_episode_time_out
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


def _plan(
    scenario: Scenario,
    actual_start_position: Position,
    mapped_start: tuple[int, int],
    pedestrian: Pedestrian,
    max_time_seconds: float,
) -> RobustSpaceTimePlanningResult:
    return robust_space_time_social_astar(
        build_scenario_grid(scenario),
        actual_start_position,
        mapped_start,
        scenario.goal,
        pedestrian.position,
        pedestrian.velocity,
        pedestrian.target_position,
        SOCIAL_DISTANCE,
        SOCIAL_WEIGHT,
        HUMAN_COLLISION_DISTANCE,
        scenario.grid_scale,
        ROBOT_SPEED,
        max_time_seconds,
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
    pedestrian: Pedestrian,
) -> SpaceTimePlanningCall:
    diagnostic_start = (
        planning_result.bridge.target_cell
        if planning_result.bridge is not None
        else mapped_start
    )
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
        pedestrian_position=pedestrian.position,
        pedestrian_velocity=pedestrian.velocity,
        pedestrian_target=pedestrian.target_position,
        grid_scale=scenario.grid_scale,
        move_duration=scenario.grid_scale / ROBOT_SPEED,
        collision_distance=HUMAN_COLLISION_DISTANCE,
        planning_result=_fallback_grid_result(planning_result),
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
    """Execute the new robust method without sharing old-method control state."""
    if method != ROBUST_SPACE_TIME_METHOD:
        raise ValueError(
            f"robust space-time method must be {ROBUST_SPACE_TIME_METHOD}"
        )
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if replan_stop_steps <= 0:
        raise ValueError("replan_stop_steps must be positive")

    grid_map = build_scenario_grid(scenario)
    astar_path = astar(grid_map, scenario.start, scenario.goal)
    if astar_path is None:
        raise ValueError("scenario goal must be reachable by A*")

    start_position = grid_to_world(scenario.start, scenario.grid_scale)
    pedestrian = Pedestrian(
        scenario.pedestrian_start,
        scenario.pedestrian_target,
        scenario.pedestrian_speed,
    )
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
        starts_unsafe = hypot(
            actual_robot_position[0] - pedestrian.position[0],
            actual_robot_position[1] - pedestrian.position[1],
        ) <= HUMAN_COLLISION_DISTANCE
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
        pedestrian,
        maximum_planning_seconds,
    )
    account_planning_result(initial_result, start_position)
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
            pedestrian=pedestrian,
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
        pedestrian_id = _create_cylinder(
            pedestrian.position,
            PEDESTRIAN_RADIUS,
            PEDESTRIAN_HEIGHT,
            client_id,
        )

        robot_trajectory = [_record_position(robot_id, client_id)]
        pedestrian_trajectory = [
            _record_position(pedestrian_id, client_id)
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
        steps = 0

        def plan_is_complete() -> bool:
            return (
                current_plan is not None
                and bridge_progress >= bridge_steps
                and action_index >= len(current_plan.grid_plan.actions)
            )

        while steps < max_steps and not plan_is_complete():
            current_robot_position = robot_trajectory[-1]

            if pending_stall_event is not None:
                mapped_start = world_to_nearest_free_cell(
                    grid_map,
                    current_robot_position,
                    scenario.grid_scale,
                )
                if suppressor.should_suppress(
                    mapped_start=mapped_start,
                    robot_position=current_robot_position,
                    pedestrian_position=pedestrian.position,
                    pedestrian_velocity=pedestrian.velocity,
                    pedestrian_target=pedestrian.target_position,
                ):
                    suppressed_duplicate_replans += 1
                else:
                    replan_steps.append(steps)
                    replanned_result = _plan(
                        scenario,
                        current_robot_position,
                        mapped_start,
                        pedestrian,
                        (max_steps - steps) * SIMULATION_STEP,
                    )
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
                            stopped_streak=replan_stop_steps,
                            total_reactive_stopped_steps=(
                                reactive_stopped_steps
                            ),
                            previous_successful_plan_step=(
                                previous_successful_plan_step
                            ),
                            actual_robot_position=current_robot_position,
                            mapped_start=mapped_start,
                            pedestrian=pedestrian,
                        )
                    )
                    if replanned_result.plan is None:
                        robust_replan_failures += 1
                        spacetime_planning_failures += 1
                        assert replanned_result.failure_reason is not None
                        suppressor.record_failure(
                            mapped_start=mapped_start,
                            robot_position=current_robot_position,
                            pedestrian_position=pedestrian.position,
                            pedestrian_velocity=pedestrian.velocity,
                            pedestrian_target=pedestrian.target_position,
                            failure_reason=replanned_result.failure_reason,
                        )
                    else:
                        robust_replan_successes += 1
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

            robot_position = current_robot_position
            intentional_wait = False
            if current_plan is None:
                speed_scale = 0.0
            elif bridge_progress < bridge_steps:
                intended_motion = (
                    current_plan.bridge.target_position[0]
                    - current_robot_position[0],
                    current_plan.bridge.target_position[1]
                    - current_robot_position[1],
                )
                speed_scale = compute_directional_speed_scale(
                    current_robot_position,
                    pedestrian.position,
                    intended_motion,
                    STOP_DISTANCE,
                    SLOW_DISTANCE,
                    ESCAPE_SPEED_SCALE,
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
                    speed_scale = compute_directional_speed_scale(
                        current_robot_position,
                        pedestrian.position,
                        intended_motion,
                        STOP_DISTANCE,
                        SLOW_DISTANCE,
                        ESCAPE_SPEED_SCALE,
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
            p.resetBasePositionAndOrientation(
                robot_id,
                (*robot_position, ROBOT_HEIGHT / 2 + 0.01),
                (0.0, 0.0, 0.0, 1.0),
                physicsClientId=client_id,
            )
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
            pedestrian_trajectory.append(
                _record_position(pedestrian_id, client_id)
            )
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
            [pedestrian_trajectory],
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
        robust_failure_reason = _classify_robust_episode_failure(
            result,
            robust_planning_failure_reasons,
            progress_stall_events,
            robust_replan_successes,
        )
        trace = EpisodeTrace(
            planned_path=planned_path,
            speed_scales=tuple(speed_scales),
            timed_out=did_episode_time_out(
                steps=steps,
                max_steps=max_steps,
                path_completed=path_completed,
            ),
            final_robot_position=robot_trajectory[-1],
            final_pedestrian_position=pedestrian_trajectory[-1],
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
        )
        return result, trace
    finally:
        p.disconnect(client_id)
