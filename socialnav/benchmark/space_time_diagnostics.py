"""Pure diagnostics for space-time planning calls and repeated failures."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from socialnav.env.demo_map import grid_to_world
from socialnav.env.grid_map import Coordinate
from socialnav.metrics import Position
from socialnav.planners.pedestrian_prediction import (
    PedestrianPredictionState,
    predict_pedestrian_position_at_time,
)
from socialnav.planners.space_time_planner import _collect_pedestrian_states
from socialnav.planners.space_time_planner import (
    SpaceTimeAction,
    SpaceTimeActionSafety,
    SpaceTimeFailureReason,
    SpaceTimePlanningResult,
    State,
)

_TARGET_TOLERANCE = 1e-12
REPEATED_STATE_QUANTIZATION = 1e-3


@dataclass(frozen=True)
class SpaceTimePlanningCall:
    """Planner evidence joined with the runner state at the call site."""

    scenario_id: str
    call_index: int
    is_initial_plan: bool
    simulation_step: int
    simulated_episode_time: float
    remaining_episode_time: float
    stopped_streak: int
    total_reactive_stopped_steps: int
    previous_successful_plan_step: int | None
    actual_robot_world_position: Position
    mapped_robot_grid_cell: Coordinate
    mapped_cell_world_center: Position
    actual_to_mapped_center_distance: float
    pedestrian_position: Position
    pedestrian_velocity: Position
    pedestrian_target: Position
    actual_robot_pedestrian_distance: float
    mapped_center_to_pedestrian_distance: float
    pedestrian_at_target: bool
    mapped_pose_safe_first_actions: tuple[SpaceTimeAction, ...]
    actual_pose_safe_first_actions: tuple[SpaceTimeAction, ...]
    mapped_start_artifact: bool
    failure_reason: SpaceTimeFailureReason | None
    first_action_safety: tuple[SpaceTimeActionSafety, ...]
    expanded_states: int
    generated_states: int
    maximum_time_index_reached: int
    planning_horizon_reached: bool
    open_set_exhausted: bool
    goal_reached: bool
    returned_path_length: int | None
    planned_wait_count: int
    returned_actions: tuple[SpaceTimeAction, ...]
    returned_timed_states: tuple[State, ...]
    pedestrian_count: int = 1


@dataclass(frozen=True)
class RepeatedFailureGroup:
    """A deterministic group of failed calls from effectively equal states."""

    scenario_id: str
    failure_reason: SpaceTimeFailureReason
    mapped_robot_grid_cell: Coordinate
    representative_actual_robot_position: Position
    representative_pedestrian_position: Position
    representative_pedestrian_velocity: Position
    call_count: int
    first_simulation_step: int
    last_simulation_step: int


def build_space_time_planning_call(
    *,
    scenario_id: str,
    call_index: int,
    is_initial_plan: bool,
    simulation_step: int,
    simulated_episode_time: float,
    remaining_episode_time: float,
    stopped_streak: int,
    total_reactive_stopped_steps: int,
    previous_successful_plan_step: int | None,
    actual_robot_world_position: Position,
    mapped_robot_grid_cell: Coordinate,
    pedestrian_position: Position,
    pedestrian_velocity: Position,
    pedestrian_target: Position,
    grid_scale: float,
    move_duration: float,
    collision_distance: float,
    planning_result: SpaceTimePlanningResult,
    additional_pedestrians: tuple[PedestrianPredictionState, ...] = (),
    pedestrian_count: int | None = None,
) -> SpaceTimePlanningCall:
    """Attach continuous-pose and mapping evidence to a planner result."""
    mapped_center = grid_to_world(mapped_robot_grid_cell, grid_scale)
    actual_to_center = hypot(
        actual_robot_world_position[0] - mapped_center[0],
        actual_robot_world_position[1] - mapped_center[1],
    )
    actual_human_distance = hypot(
        actual_robot_world_position[0] - pedestrian_position[0],
        actual_robot_world_position[1] - pedestrian_position[1],
    )
    mapped_human_distance = hypot(
        mapped_center[0] - pedestrian_position[0],
        mapped_center[1] - pedestrian_position[1],
    )
    mapped_safe_actions = tuple(
        detail.action
        for detail in planning_result.first_action_safety
        if detail.rejection_reason is None
    )
    actual_safe_actions = tuple(
        detail.action
        for detail in planning_result.first_action_safety
        if (
            detail.inside_map
            and detail.free
            and _actual_pose_action_is_safe(
                detail.action,
                detail.destination,
                actual_robot_world_position=actual_robot_world_position,
                pedestrian_position=pedestrian_position,
                pedestrian_velocity=pedestrian_velocity,
                pedestrian_target=pedestrian_target,
                grid_scale=grid_scale,
                move_duration=move_duration,
                collision_distance=collision_distance,
                additional_pedestrians=additional_pedestrians,
            )
        )
    )
    resolved_pedestrian_count = (
        (1 if pedestrian_position is not None else 0)
        + len(additional_pedestrians)
        if pedestrian_count is None
        else pedestrian_count
    )
    mapped_start_artifact = (
        planning_result.plan is None
        and not mapped_safe_actions
        and bool(actual_safe_actions)
    )
    statistics = planning_result.statistics
    plan = planning_result.plan
    return SpaceTimePlanningCall(
        scenario_id=scenario_id,
        call_index=call_index,
        is_initial_plan=is_initial_plan,
        simulation_step=simulation_step,
        simulated_episode_time=simulated_episode_time,
        remaining_episode_time=remaining_episode_time,
        stopped_streak=stopped_streak,
        total_reactive_stopped_steps=total_reactive_stopped_steps,
        previous_successful_plan_step=previous_successful_plan_step,
        actual_robot_world_position=actual_robot_world_position,
        mapped_robot_grid_cell=mapped_robot_grid_cell,
        mapped_cell_world_center=mapped_center,
        actual_to_mapped_center_distance=actual_to_center,
        pedestrian_position=pedestrian_position,
        pedestrian_velocity=pedestrian_velocity,
        pedestrian_target=pedestrian_target,
        actual_robot_pedestrian_distance=actual_human_distance,
        mapped_center_to_pedestrian_distance=mapped_human_distance,
        pedestrian_at_target=(
            hypot(
                pedestrian_position[0] - pedestrian_target[0],
                pedestrian_position[1] - pedestrian_target[1],
            )
            <= _TARGET_TOLERANCE
        ),
        mapped_pose_safe_first_actions=mapped_safe_actions,
        actual_pose_safe_first_actions=actual_safe_actions,
        mapped_start_artifact=mapped_start_artifact,
        failure_reason=planning_result.failure_reason,
        first_action_safety=planning_result.first_action_safety,
        expanded_states=statistics.expanded_states,
        generated_states=statistics.generated_states,
        maximum_time_index_reached=statistics.maximum_time_index_reached,
        planning_horizon_reached=statistics.planning_horizon_reached,
        open_set_exhausted=statistics.open_set_exhausted,
        goal_reached=statistics.goal_reached,
        returned_path_length=statistics.returned_path_length,
        planned_wait_count=statistics.planned_wait_count,
        returned_actions=() if plan is None else plan.actions,
        returned_timed_states=() if plan is None else plan.timed_states,
        pedestrian_count=resolved_pedestrian_count,
    )


def _actual_pose_action_is_safe(
    action: SpaceTimeAction,
    destination: Coordinate,
    *,
    actual_robot_world_position: Position,
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    grid_scale: float,
    move_duration: float,
    collision_distance: float,
    additional_pedestrians: tuple[PedestrianPredictionState, ...] = (),
) -> bool:
    pedestrians = _collect_pedestrian_states(
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        additional_pedestrians,
    )
    end_position = (
        actual_robot_world_position
        if action == "WAIT"
        else grid_to_world(destination, grid_scale)
    )
    for fraction in (0.0, 0.5, 1.0):
        robot_position = (
            actual_robot_world_position[0]
            + (end_position[0] - actual_robot_world_position[0]) * fraction,
            actual_robot_world_position[1]
            + (end_position[1] - actual_robot_world_position[1]) * fraction,
        )
        for position, velocity, target in pedestrians:
            pedestrian_at_time = predict_pedestrian_position_at_time(
                position,
                velocity,
                move_duration * fraction,
                target=target,
            )
            if hypot(
                robot_position[0] - pedestrian_at_time[0],
                robot_position[1] - pedestrian_at_time[1],
            ) <= collision_distance:
                return False
    return True


def group_repeated_failed_planning_calls(
    calls: tuple[SpaceTimePlanningCall, ...] | list[SpaceTimePlanningCall],
    *,
    quantization: float = REPEATED_STATE_QUANTIZATION,
) -> tuple[RepeatedFailureGroup, ...]:
    """Group failed calls by near-identical continuous and discrete state."""
    if quantization <= 0.0:
        raise ValueError("quantization must be positive")
    grouped: dict[tuple[object, ...], list[SpaceTimePlanningCall]] = {}
    for call in calls:
        if call.failure_reason is None:
            continue
        key = (
            call.scenario_id,
            call.failure_reason,
            call.mapped_robot_grid_cell,
            _quantize_position(call.actual_robot_world_position, quantization),
            _quantize_position(call.pedestrian_position, quantization),
            _quantize_position(call.pedestrian_velocity, quantization),
        )
        grouped.setdefault(key, []).append(call)

    results = []
    for members in grouped.values():
        ordered = sorted(members, key=lambda call: call.simulation_step)
        representative = ordered[0]
        assert representative.failure_reason is not None
        results.append(
            RepeatedFailureGroup(
                scenario_id=representative.scenario_id,
                failure_reason=representative.failure_reason,
                mapped_robot_grid_cell=(
                    representative.mapped_robot_grid_cell
                ),
                representative_actual_robot_position=(
                    representative.actual_robot_world_position
                ),
                representative_pedestrian_position=(
                    representative.pedestrian_position
                ),
                representative_pedestrian_velocity=(
                    representative.pedestrian_velocity
                ),
                call_count=len(ordered),
                first_simulation_step=ordered[0].simulation_step,
                last_simulation_step=ordered[-1].simulation_step,
            )
        )
    return tuple(
        sorted(
            results,
            key=lambda group: (
                group.scenario_id,
                group.first_simulation_step,
                group.mapped_robot_grid_cell,
                group.failure_reason,
            ),
        )
    )


def _quantize_position(
    position: Position,
    quantization: float,
) -> tuple[int, int]:
    return (
        round(position[0] / quantization),
        round(position[1] / quantization),
    )
