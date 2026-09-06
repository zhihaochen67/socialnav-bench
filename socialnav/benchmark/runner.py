"""Faithful headless execution of one benchmark episode."""

from math import floor, hypot, inf

import pybullet as p

from socialnav.env.demo_map import (
    SOCIAL_DISTANCE,
    SOCIAL_WEIGHT,
    grid_to_world,
    interpolate_path,
)
from socialnav.env.pedestrian import (
    Pedestrian,
    compute_pedestrian_velocity,
)
from socialnav.planners.pedestrian_prediction import (
    PedestrianPredictionState,
)
from socialnav.env.world import (
    GOAL_TOLERANCE,
    HUMAN_COLLISION_DISTANCE,
    OBSTACLE_HALF_EXTENT,
    OBSTACLE_HEIGHT,
    PEDESTRIAN_HEIGHT,
    PEDESTRIAN_RADIUS,
    ROBOT_HEIGHT,
    ROBOT_RADIUS,
    SIMULATION_STEP,
    SLOW_DISTANCE,
    STEPS_PER_CELL,
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
from socialnav.planners.clearance_recovery import (
    find_clearance_recovery_path,
    is_multi_clearance_safe_motion,
)
from socialnav.planners.directional_avoidance import (
    ESCAPE_SPEED_SCALE,
    compute_directional_speed_scale,
)
from socialnav.planners.dynamic_avoidance import compute_multi_speed_scale
from socialnav.planners.predictive_social_planner import (
    predictive_social_astar,
)
from socialnav.planners.social_planner import social_astar

from .diagnostics import EpisodeTrace, did_episode_time_out
from .replanning import (
    REPLAN_STOP_STEPS,
    SustainedStopReplanPolicy,
    interpolate_replanned_route,
    world_to_nearest_free_cell,
)
from .robust_space_time_runner import (
    ROBUST_SPACE_TIME_METHOD,
    run_robust_space_time_episode_with_trace,
)
from .scenario import PedestrianSpec, Scenario, build_scenario_grid
from .space_time_runner import (
    SPACE_TIME_METHODS,
    run_space_time_episode_with_trace,
)

SUPPORTED_METHODS = (
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
    "social_spacetime_robust",
)
MAX_EPISODE_SECONDS = 20.0
MAX_EPISODE_STEPS = int(MAX_EPISODE_SECONDS / SIMULATION_STEP)


def _prediction_states(
    specs: tuple[PedestrianSpec, ...],
) -> tuple[PedestrianPredictionState, ...]:
    """Initial prediction state for every pedestrian beyond the first."""
    return tuple(
        (
            spec.start,
            compute_pedestrian_velocity(
                spec.start,
                spec.target,
                spec.speed,
            ),
            spec.target,
        )
        for spec in specs
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


def _path_position(
    positions: list[Position],
    progress: float,
) -> Position:
    lower_index = floor(progress)
    if lower_index >= len(positions) - 1:
        return positions[-1]

    fraction = progress - lower_index
    start_x, start_y = positions[lower_index]
    end_x, end_y = positions[lower_index + 1]
    return (
        start_x + (end_x - start_x) * fraction,
        start_y + (end_y - start_y) * fraction,
    )


def _next_motion_vector(
    positions: list[Position],
    progress: float,
    current_position: Position,
) -> Position:
    """Return motion from the robot's actual position to its next route point."""
    target_index = min(floor(progress) + 1, len(positions) - 1)
    target = positions[target_index]
    return (
        target[0] - current_position[0],
        target[1] - current_position[1],
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
        key=lambda pedestrian: hypot(
            robot_position[0] - pedestrian.position[0],
            robot_position[1] - pedestrian.position[1],
        ),
    )


def _blocking_human_indices(
    pedestrians: list[Pedestrian],
    robot_position: Position,
) -> tuple[int, ...]:
    """Indices of pedestrians inside the hard-stop region of the robot."""
    return tuple(
        index
        for index, pedestrian in enumerate(pedestrians)
        if hypot(
            robot_position[0] - pedestrian.position[0],
            robot_position[1] - pedestrian.position[1],
        )
        <= STOP_DISTANCE
    )


def run_episode(
    scenario: Scenario,
    method: str,
    *,
    max_steps: int = MAX_EPISODE_STEPS,
    replan_stop_steps: int = REPLAN_STOP_STEPS,
) -> EpisodeResult:
    """Run one deterministic benchmark episode in PyBullet DIRECT mode."""
    result, _ = run_episode_with_trace(
        scenario,
        method,
        max_steps=max_steps,
        replan_stop_steps=replan_stop_steps,
    )
    return result
def run_episode_with_trace(
    scenario: Scenario,
    method: str,
    *,
    max_steps: int = MAX_EPISODE_STEPS,
    replan_stop_steps: int = REPLAN_STOP_STEPS,
) -> tuple[EpisodeResult, EpisodeTrace]:
    """Run an episode and return metrics plus diagnostic runner state."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"method must be one of {', '.join(SUPPORTED_METHODS)}"
        )
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if (
        method
        in (
            "social_replan",
            "social_replan_escape",
            "social_replan_recovery",
            "social_predictive_replan",
            "social_spacetime_replan",
            "social_spacetime_robust",
        )
        and replan_stop_steps <= 0
    ):
        raise ValueError("replan_stop_steps must be positive")

    if method == ROBUST_SPACE_TIME_METHOD:
        return run_robust_space_time_episode_with_trace(
            scenario,
            method,
            max_steps=max_steps,
            replan_stop_steps=replan_stop_steps,
        )

    if method in SPACE_TIME_METHODS:
        return run_space_time_episode_with_trace(
            scenario,
            method,
            max_steps=max_steps,
            replan_stop_steps=replan_stop_steps,
        )

    grid_map = build_scenario_grid(scenario)
    astar_path = astar(grid_map, scenario.start, scenario.goal)
    if astar_path is None:
        raise ValueError("scenario goal must be reachable by A*")

    pedestrian_specs = scenario.pedestrians
    pedestrian_positions_at_start = [
        spec.start for spec in pedestrian_specs
    ]

    if method in (
        "social",
        "social_replan",
        "social_replan_escape",
        "social_replan_recovery",
    ):
        selected_path = social_astar(
            grid_map,
            scenario.start,
            scenario.goal,
            pedestrian_positions=pedestrian_positions_at_start,
            social_distance=SOCIAL_DISTANCE,
            social_weight=SOCIAL_WEIGHT,
            grid_scale=scenario.grid_scale,
        )
        if selected_path is None:
            raise RuntimeError("social A* could not find a scenario path")
    elif method in ("social_predictive", "social_predictive_replan"):
        if pedestrian_specs:
            first_spec = pedestrian_specs[0]
            selected_path = predictive_social_astar(
                grid_map,
                scenario.start,
                scenario.goal,
                first_spec.start,
                compute_pedestrian_velocity(
                    first_spec.start,
                    first_spec.target,
                    first_spec.speed,
                ),
                SOCIAL_DISTANCE,
                SOCIAL_WEIGHT,
                scenario.grid_scale,
                pedestrian_target=first_spec.target,
                additional_pedestrians=_prediction_states(
                    pedestrian_specs[1:]
                ),
            )
        else:
            selected_path = astar_path
        if selected_path is None:
            raise RuntimeError(
                "predictive social A* could not find a scenario path"
            )
    else:
        selected_path = astar_path

    path_positions = interpolate_path(
        selected_path,
        steps_per_cell=STEPS_PER_CELL,
        cell_size=scenario.grid_scale,
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
            path_positions[0],
            ROBOT_RADIUS,
            ROBOT_HEIGHT,
            client_id,
        )
        pedestrians = [
            Pedestrian(spec.start, spec.target, spec.speed)
            for spec in pedestrian_specs
        ]
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
        progress = 0.0
        last_position_index = len(path_positions) - 1
        steps = 0
        speed_scales: list[float] = []
        replan_policy = (
            SustainedStopReplanPolicy(replan_stop_steps)
            if method
            in (
                "social_replan",
                "social_replan_escape",
                "social_replan_recovery",
                "social_predictive_replan",
            )
            else None
        )
        replan_steps: list[int] = []
        successful_replans = 0
        failed_replans = 0
        recovery_active = False
        recovery_trigger_steps: list[int] = []
        successful_recoveries = 0
        failed_recoveries = 0
        recovery_path_lengths: list[int] = []

        while steps < max_steps and (
            recovery_active or progress < last_position_index
        ):
            current_robot_position = robot_trajectory[-1]
            closest = _closest_pedestrian(
                pedestrians,
                current_robot_position,
            )
            blocking_distance = (
                inf
                if closest is None
                else hypot(
                    current_robot_position[0] - closest.position[0],
                    current_robot_position[1] - closest.position[1],
                )
            )
            if recovery_active and (
                progress >= last_position_index
                or blocking_distance >= SLOW_DISTANCE
            ):
                recovery_active = False
                assert replan_policy is not None
                replan_policy.consecutive_stopped_steps = 0
                replan_steps.append(steps + 1)
                replan_start = world_to_nearest_free_cell(
                    grid_map,
                    current_robot_position,
                    scenario.grid_scale,
                )
                replanned_path = social_astar(
                    grid_map,
                    replan_start,
                    scenario.goal,
                    pedestrian_positions=[
                        pedestrian.position for pedestrian in pedestrians
                    ],
                    social_distance=SOCIAL_DISTANCE,
                    social_weight=SOCIAL_WEIGHT,
                    grid_scale=scenario.grid_scale,
                )
                if replanned_path is None:
                    failed_replans += 1
                    path_positions = [
                        current_robot_position,
                        current_robot_position,
                    ]
                else:
                    successful_replans += 1
                    selected_path = replanned_path
                    path_positions = interpolate_replanned_route(
                        current_robot_position,
                        selected_path,
                        scenario.grid_scale,
                        steps_per_cell=STEPS_PER_CELL,
                    )
                progress = 0.0
                last_position_index = len(path_positions) - 1

            if method == "astar":
                speed_scale = 1.0
            elif method in (
                "social_replan_escape",
                "social_replan_recovery",
            ):
                intended_motion = _next_motion_vector(
                    path_positions,
                    progress,
                    current_robot_position,
                )
                recovery_motion_is_safe = (
                    not recovery_active
                    or is_multi_clearance_safe_motion(
                        current_robot_position,
                        [
                            pedestrian.position
                            for pedestrian in pedestrians
                        ],
                        intended_motion,
                    )
                )
                if recovery_motion_is_safe:
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
                else:
                    speed_scale = 0.0
            else:
                speed_scale = compute_multi_speed_scale(
                    current_robot_position,
                    [
                        pedestrian.position
                        for pedestrian in pedestrians
                    ],
                    STOP_DISTANCE,
                    SLOW_DISTANCE,
                )

            speed_scales.append(speed_scale)
            if replan_policy is not None and replan_policy.observe(speed_scale):
                replan_steps.append(steps + 1)
                current_robot_position = robot_trajectory[-1]
                replan_start = world_to_nearest_free_cell(
                    grid_map,
                    current_robot_position,
                    scenario.grid_scale,
                )
                if method == "social_predictive_replan":
                    if pedestrians:
                        closest = _closest_pedestrian(
                            pedestrians,
                            current_robot_position,
                        )
                        assert closest is not None
                        replanned_path = predictive_social_astar(
                            grid_map,
                            replan_start,
                            scenario.goal,
                            closest.position,
                            closest.velocity,
                            SOCIAL_DISTANCE,
                            SOCIAL_WEIGHT,
                            scenario.grid_scale,
                            pedestrian_target=closest.target_position,
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
                    else:
                        replanned_path = astar(
                            grid_map,
                            replan_start,
                            scenario.goal,
                        )
                else:
                    replanned_path = social_astar(
                        grid_map,
                        replan_start,
                        scenario.goal,
                        pedestrian_positions=[
                            pedestrian.position
                            for pedestrian in pedestrians
                        ],
                        social_distance=SOCIAL_DISTANCE,
                        social_weight=SOCIAL_WEIGHT,
                        grid_scale=scenario.grid_scale,
                    )
                if replanned_path is None:
                    failed_replans += 1
                else:
                    successful_replans += 1
                    selected_path = replanned_path
                    path_positions = interpolate_replanned_route(
                        current_robot_position,
                        selected_path,
                        scenario.grid_scale,
                        steps_per_cell=STEPS_PER_CELL,
                    )
                    progress = 0.0
                    last_position_index = len(path_positions) - 1
                    recovery_active = False
                if method == "social_replan_recovery":
                    if closest is None:
                        route_start_scale = 1.0
                    else:
                        route_start_scale = compute_directional_speed_scale(
                            current_robot_position,
                            closest.position,
                            _next_motion_vector(
                                path_positions,
                                progress,
                                current_robot_position,
                            ),
                            STOP_DISTANCE,
                            SLOW_DISTANCE,
                            ESCAPE_SPEED_SCALE,
                            additional_pedestrian_positions=[
                                pedestrian.position
                                for pedestrian in pedestrians
                                if pedestrian is not closest
                            ],
                        )
                    if route_start_scale == 0.0:
                        recovery_trigger_steps.append(steps + 1)
                        if closest is None:
                            failed_recoveries += 1
                        else:
                            recovery_path = find_clearance_recovery_path(
                                grid_map,
                                replan_start,
                                current_robot_position,
                                closest.position,
                                scenario.grid_scale,
                                SLOW_DISTANCE,
                            )
                            if recovery_path is None:
                                failed_recoveries += 1
                            else:
                                successful_recoveries += 1
                                recovery_path_lengths.append(
                                    len(recovery_path)
                                )
                                if recovery_path:
                                    path_positions = (
                                        interpolate_replanned_route(
                                            current_robot_position,
                                            recovery_path,
                                            scenario.grid_scale,
                                            steps_per_cell=STEPS_PER_CELL,
                                        )
                                    )
                                    progress = 0.0
                                    last_position_index = (
                                        len(path_positions) - 1
                                    )
                                    recovery_active = True

            progress = min(
                progress + speed_scale,
                float(last_position_index),
            )
            robot_position = _path_position(path_positions, progress)
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
        final_robot_position = robot_trajectory[-1]
        closest_final = _closest_pedestrian(pedestrians, final_robot_position)
        trace = EpisodeTrace(
            planned_path=tuple(
                grid_to_world(coordinate, scenario.grid_scale)
                for coordinate in selected_path
            ),
            speed_scales=tuple(speed_scales),
            timed_out=did_episode_time_out(
                steps=steps,
                max_steps=max_steps,
                path_completed=(
                    not recovery_active and progress >= last_position_index
                ),
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
            recovery_count=len(recovery_trigger_steps),
            recovery_trigger_steps=tuple(recovery_trigger_steps),
            successful_recoveries=successful_recoveries,
            failed_recoveries=failed_recoveries,
            recovery_path_lengths=tuple(recovery_path_lengths),
            pedestrian_count=len(pedestrian_specs),
            initial_pedestrian_positions=tuple(
                spec.start for spec in pedestrian_specs
            ),
            initial_pedestrian_velocities=tuple(
                compute_pedestrian_velocity(
                    spec.start,
                    spec.target,
                    spec.speed,
                )
                for spec in pedestrian_specs
            ),
            pedestrian_targets=tuple(
                spec.target for spec in pedestrian_specs
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
            blocking_human_indices=_blocking_human_indices(
                pedestrians,
                final_robot_position,
            ),
        )
        return result, trace
    finally:
        p.disconnect(client_id)