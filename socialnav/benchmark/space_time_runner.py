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
from socialnav.metrics import Position, compute_path_length
from socialnav.planners.astar import astar
from socialnav.planners.dynamic_avoidance import compute_speed_scale
from socialnav.planners.space_time_planner import (
    SpaceTimePlan,
    duration_to_simulation_steps,
    space_time_social_astar,
)

from .diagnostics import EpisodeTrace, did_episode_time_out
from .replanning import SustainedStopReplanPolicy, world_to_nearest_free_cell
from .scenario import Scenario, build_scenario_grid

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


def _plan(
    scenario: Scenario,
    start: tuple[int, int],
    pedestrian: Pedestrian,
    max_time_seconds: float,
) -> SpaceTimePlan | None:
    grid_map = build_scenario_grid(scenario)
    return space_time_social_astar(
        grid_map,
        start,
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

    pedestrian = Pedestrian(
        scenario.pedestrian_start,
        scenario.pedestrian_target,
        scenario.pedestrian_speed,
    )
    maximum_planning_seconds = max_steps * SIMULATION_STEP
    current_plan = _plan(
        scenario,
        scenario.start,
        pedestrian,
        maximum_planning_seconds,
    )
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
                    speed_scale = compute_speed_scale(
                        current_robot_position,
                        pedestrian.position,
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
                        replanned = _plan(
                            scenario,
                            replan_start,
                            pedestrian,
                            maximum_planning_seconds,
                        )
                        if replanned is None:
                            failed_replans += 1
                            spacetime_planning_failures += 1
                        else:
                            successful_replans += 1
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
            pedestrian_trajectory.append(
                _record_position(pedestrian_id, client_id)
            )

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
        planned_path = (
            (start_position,)
            if selected_plan is None
            else tuple(
                grid_to_world(coordinate, scenario.grid_scale)
                for coordinate in selected_plan.spatial_path
            )
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
            successful_replans=successful_replans,
            failed_replans=failed_replans,
            planned_wait_actions=planned_wait_actions,
            executed_wait_actions=executed_wait_actions,
            planned_move_actions=planned_move_actions,
            spacetime_plan_count=spacetime_plan_count,
            spacetime_planning_failures=spacetime_planning_failures,
            total_intentional_wait_steps=total_intentional_wait_steps,
            reactive_stopped_steps=reactive_stopped_steps,
        )
        return result, trace
    finally:
        p.disconnect(client_id)
