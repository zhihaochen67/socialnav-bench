"""Headless execution for explicit-action space-time social plans."""

from __future__ import annotations

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
    OBSTACLE_HEIGHT,
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
from socialnav.planners.dynamic_avoidance import compute_multi_speed_scale
from socialnav.planners.space_time_planner import (
    SpaceTimePlan,
    SpaceTimePlanningResult,
    SpaceTimeSearchStatistics,
    duration_to_simulation_steps,
    space_time_social_astar,
)

from .diagnostics import EpisodeTrace, did_episode_time_out
from .replanning import SustainedStopReplanPolicy, world_to_nearest_free_cell
from .scenario import Scenario, build_scenario_grid
from .space_time_diagnostics import (
    SpaceTimePlanningCall,
    build_space_time_planning_call,
)

SPACE_TIME_METHODS = (
    "social_spacetime",
    "social_spacetime_replan",
)


def _create_cylinder(
    position: Position,
    radius: float,
    height: float,
    client_id: int,
) -> int:
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height,
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=-1,
        basePosition=(position[0], position[1], height / 2 + 0.01),
        physicsClientId=client_id,
    )


def _create_obstacles(scenario: Scenario, client_id: int) -> None:
    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=(
            OBSTACLE_HALF_EXTENT,
            OBSTACLE_HALF_EXTENT,
            OBSTACLE_HEIGHT / 2,
        ),
        physicsClientId=client_id,
    )
    for obstacle in scenario.obstacle_cells:
        x, y = grid_to_world(obstacle, scenario.grid_scale)
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=-1,
            basePosition=(x, y, OBSTACLE_HEIGHT / 2),
            physicsClientId=client_id,
        )


def _record_position(body_id: int, client_id: int) -> Position:
    position, _ = p.getBasePositionAndOrientation(
        body_id,
        physicsClientId=client_id,
    )
    return position[0], position[1]


def _closest_pedestrian(
    pedestrians: list[Pedestrian],
    robot_position: Position,
) -> Pedestrian | None:
    """Return the pedestrian nearest the robot, with stable index-order ties."""
    if not pedestrians:
        return None
    return min(
        pedestrians,
        key=lambda pedestrian: (
            (pedestrian.position[0] - robot_position[0]) ** 2
            + (pedestrian.position[1] - robot_position[1]) ** 2
        ),
    )


def _plan(
    scenario: Scenario,
    start: tuple[int, int],
    pedestrians: list[Pedestrian],
    max_time_seconds: float,
) -> tuple[SpaceTimePlan | None, SpaceTimePlanningResult]:
    grid_map = build_scenario_grid(scenario)
    diagnostic_results: list[SpaceTimePlanningResult] = []
    closest = _closest_pedestrian(
        pedestrians,
        grid_to_world(start, scenario.grid_scale),
    )
    if closest is None:
        plan = space_time_social_astar(
            grid_map,
            start,
            scenario.goal,
            None,
            (0.0, 0.0),
            None,
            SOCIAL_DISTANCE,
            SOCIAL_WEIGHT,
            HUMAN_COLLISION_DISTANCE,
            scenario.grid_scale,
            ROBOT_SPEED,
            max_time_seconds,
            diagnostic_results=diagnostic_results,
        )
    else:
        plan = space_time_social_astar(
            grid_map,
            start,
            scenario.goal,
            closest.position,
            closest.velocity,
            closest.target_position,
            SOCIAL_DISTANCE,
            SOCIAL_WEIGHT,
            HUMAN_COLLISION_DISTANCE,
            scenario.grid_scale,
            ROBOT_SPEED,
            max_time_seconds,
            diagnostic_results=diagnostic_results,
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
    planning_result = (
        diagnostic_results[0]
        if diagnostic_results
        else _fallback_planning_result(plan)
    )
    return plan, planning_result


def _fallback_planning_result(
    plan: SpaceTimePlan | None,
) -> SpaceTimePlanningResult:
    """Keep monkeypatched planner tests compatible with diagnostic tracing."""
    return SpaceTimePlanningResult(
        plan=plan,
        failure_reason=None if plan is not None else "other",
        first_action_safety=(),
        statistics=SpaceTimeSearchStatistics(
            expanded_states=0,
            generated_states=0,
            maximum_time_index_reached=(
                0
                if plan is None
                else max(state[2] for state in plan.timed_states)
            ),
            planning_horizon_reached=False,
            open_set_exhausted=plan is None,
            goal_reached=plan is not None,
            returned_path_length=None if plan is None else len(plan.actions),
            planned_wait_count=(
                0 if plan is None else plan.planned_wait_actions
            ),
        ),
    )
def _planning_call(
    *,
    scenario: Scenario,
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
    planning_result: SpaceTimePlanningResult,
) -> SpaceTimePlanningCall:
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
        mapped_robot_grid_cell=mapped_start,
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
        planning_result=planning_result,
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


def run_space_time_episode_with_trace(
    scenario: Scenario,
    method: str,
    *,
    max_steps: int,
    replan_stop_steps: int,
) -> tuple[EpisodeResult, EpisodeTrace]:
    """Execute one space-time benchmark episode without changing old methods."""
    if method not in SPACE_TIME_METHODS:
        raise ValueError(
            f"space-time method must be one of {', '.join(SPACE_TIME_METHODS)}"
        )

    grid_map = build_scenario_grid(scenario)
    astar_path = astar(grid_map, scenario.start, scenario.goal)
    if astar_path is None:
        raise ValueError("scenario goal must be reachable by A*")

    pedestrians = [
        Pedestrian(spec.start, spec.target, spec.speed)
        for spec in scenario.pedestrians
    ]
    maximum_planning_seconds = max_steps * SIMULATION_STEP
    start_position = grid_to_world(scenario.start, scenario.grid_scale)
    current_plan, initial_planning_result = _plan(
        scenario,
        scenario.start,
        pedestrians,
        maximum_planning_seconds,
    )
    final_planning_result = initial_planning_result
    planning_calls: list[SpaceTimePlanningCall] = [
        _planning_call(
            scenario=scenario,
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
            planning_result=initial_planning_result,
        )
    ]
    previous_successful_plan_step = 0 if current_plan is not None else None
    selected_plan = current_plan
    spacetime_plan_count = 1
    spacetime_planning_failures = int(current_plan is None)
    planned_wait_actions = (
        0 if current_plan is None else current_plan.planned_wait_actions
    )
    planned_move_actions = (
        0 if current_plan is None else current_plan.planned_move_actions
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
    start_position = grid_to_world(scenario.start, scenario.grid_scale)

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
        replan_policy = (
            SustainedStopReplanPolicy(replan_stop_steps)
            if method == "social_spacetime_replan"
            else None
        )
        replan_steps: list[int] = []
        successful_replans = 0
        failed_replans = 0
        executed_wait_actions = 0
        total_intentional_wait_steps = 0
        reactive_stopped_steps = 0
        action_index = 0
        action_progress = 0.0
        action_start_position: Position | None = None
        steps = 0

        def plan_is_complete() -> bool:
            return (
                current_plan is not None
                and action_index >= len(current_plan.actions)
            )

        while steps < max_steps and (
            current_plan is None or not plan_is_complete()
        ):
            current_robot_position = robot_trajectory[-1]
            robot_position = current_robot_position

            if current_plan is None:
                speed_scale = 1.0
            else:
                action = current_plan.actions[action_index]
                if action_start_position is None:
                    action_start_position = current_robot_position

                if action == "WAIT":
                    speed_scale = 1.0
                    total_intentional_wait_steps += 1
                    action_progress += 1.0
                    if replan_policy is not None:
                        replan_policy.consecutive_stopped_steps = 0
                    if action_progress >= action_steps:
                        executed_wait_actions += 1
                        action_index += 1
                        action_progress = 0.0
                        action_start_position = None
                else:
                    next_state = current_plan.timed_states[action_index + 1]
                    target_position = grid_to_world(
                        (next_state[0], next_state[1]),
                        scenario.grid_scale,
                    )
                    speed_scale = compute_multi_speed_scale(
                        current_robot_position,
                        [
                            pedestrian.position
                            for pedestrian in pedestrians
                        ],
                        STOP_DISTANCE,
                        SLOW_DISTANCE,
                    )
                    if speed_scale == 0.0:
                        reactive_stopped_steps += 1

                    replan_triggered = (
                        replan_policy is not None
                        and replan_policy.observe(speed_scale)
                    )
                    if replan_triggered:
                        replan_steps.append(steps + 1)
                        replan_start = world_to_nearest_free_cell(
                            grid_map,
                            current_robot_position,
                            scenario.grid_scale,
                        )
                        spacetime_plan_count += 1
                        replanned, planning_result = _plan(
                            scenario,
                            replan_start,
                            pedestrians,
                            maximum_planning_seconds,
                        )
                        final_planning_result = planning_result
                        call_step = steps + 1
                        planning_calls.append(
                            _planning_call(
                                scenario=scenario,
                                call_index=len(planning_calls),
                                is_initial_plan=False,
                                simulation_step=call_step,
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
                                actual_robot_position=(
                                    current_robot_position
                                ),
                                mapped_start=replan_start,
                                pedestrians=pedestrians,
                                planning_result=planning_result,
                            )
                        )
                        if replanned is None:
                            failed_replans += 1
                            spacetime_planning_failures += 1
                        else:
                            successful_replans += 1
                            previous_successful_plan_step = call_step
                            current_plan = replanned
                            selected_plan = replanned
                            planned_wait_actions += (
                                replanned.planned_wait_actions
                            )
                            planned_move_actions += (
                                replanned.planned_move_actions
                            )
                            action_index = 0
                            action_progress = 0.0
                            action_start_position = None
                    else:
                        action_progress = min(
                            action_progress + speed_scale,
                            float(action_steps),
                        )
                        assert action_start_position is not None
                        fraction = action_progress / action_steps
                        robot_position = (
                            action_start_position[0]
                            + (
                                target_position[0]
                                - action_start_position[0]
                            )
                            * fraction,
                            action_start_position[1]
                            + (
                                target_position[1]
                                - action_start_position[1]
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
            robot_trajectory.append(_record_position(robot_id, client_id))
            for trajectory, pedestrian_id in zip(
                pedestrian_trajectories,
                pedestrian_ids,
            ):
                trajectory.append(_record_position(pedestrian_id, client_id))
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
            planned_path = tuple(
                grid_to_world(coordinate, scenario.grid_scale)
                for coordinate in selected_plan.spatial_path
            )
        final_robot_position = robot_trajectory[-1]
        closest_final = _closest_pedestrian(
            pedestrians,
            final_robot_position,
        )
        minimum_predicted_separation = min(
            (
                detail.minimum_predicted_separation
                for detail in final_planning_result.first_action_safety
            ),
            default=None,
        )
        trace = EpisodeTrace(
            planned_path=planned_path,
            speed_scales=tuple(speed_scales),
            timed_out=did_episode_time_out(
                steps=steps,
                max_steps=max_steps,
                path_completed=path_completed,
            ),
            final_robot_position=final_robot_position,
            final_pedestrian_position=(
                closest_final.position
                if closest_final is not None
                else (0.0, 0.0)
            ),
            max_steps=max_steps,
            replan_count=len(replan_steps),
            replan_steps=tuple(replan_steps),
            successful_replans=successful_replans,
            failed_replans=failed_replans,
            planned_wait_actions=planned_wait_actions,
            executed_wait_actions=executed_wait_actions,
            planned_move_actions=planned_move_actions,
            spacetime_plan_count=spacetime_plan_count,
            spacetime_planning_failures=spacetime_planning_failures,
            total_intentional_wait_steps=total_intentional_wait_steps,
            space_time_planning_calls=tuple(planning_calls),
            reactive_stopped_steps=reactive_stopped_steps,
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
                if (
                    (pedestrian.position[0] - final_robot_position[0]) ** 2
                    + (pedestrian.position[1] - final_robot_position[1]) ** 2
                )
                ** 0.5
                <= STOP_DISTANCE
            ),
            minimum_predicted_separation=minimum_predicted_separation,
        )
        return result, trace
    finally:
        p.disconnect(client_id)