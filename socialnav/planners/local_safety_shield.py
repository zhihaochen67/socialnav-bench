"""Pure predictive local-action safety for shielded execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import hypot, inf
from typing import Literal

from socialnav.env.demo_map import grid_to_world
from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.metrics import Position
from socialnav.planners.directional_avoidance import ESCAPE_SPEED_SCALE
from socialnav.planners.pedestrian_prediction import PedestrianPredictionState
from socialnav.planners.robust_space_time_planner import (
    _continuous_motion_separations,
    is_multi_collision_egress_motion_safe,
)
from socialnav.planners.space_time_planner import SpaceTimeAction

ShieldTriggerReason = Literal[
    "planned_move_predicted_unsafe",
    "stationary_wait_predicted_unsafe",
    "already_in_collision",
    "reactive_stop_with_incoming_human",
    "other",
]
ShieldRejectionReason = Literal[
    "outside_map",
    "obstacle",
    "predicted_collision",
    "non_improving_egress",
    "egress_does_not_exit_collision",
]

LOCAL_ACTIONS: tuple[tuple[SpaceTimeAction, Coordinate], ...] = (
    ("UP", (0, -1)),
    ("RIGHT", (1, 0)),
    ("DOWN", (0, 1)),
    ("LEFT", (-1, 0)),
    ("WAIT", (0, 0)),
)
LOCAL_ACTION_ORDER = {
    action: index for index, (action, _) in enumerate(LOCAL_ACTIONS)
}


@dataclass(frozen=True)
class ShieldCandidateEvaluation:
    """Safety evidence for one local action from the actual robot pose."""

    action: SpaceTimeAction
    target_cell: Coordinate
    target_position: Position
    speed_scale: float
    execution_speed: float
    distance: float
    duration: float
    static_valid: bool
    per_human_separations: tuple[tuple[float, float, float], ...]
    unsafe_human_indices: tuple[int, ...]
    minimum_predicted_separation: float
    starts_in_collision: bool
    collision_egress: bool
    safe: bool
    rejection_reason: ShieldRejectionReason | None


def evaluate_local_action(
    grid_map: GridMap,
    actual_robot_position: Position,
    mapped_start: Coordinate,
    action: SpaceTimeAction,
    pedestrian_states: Iterable[PedestrianPredictionState],
    *,
    grid_scale: float,
    robot_speed: float,
    collision_distance: float,
    move_speed_scale: float = 1.0,
    target_cell: Coordinate | None = None,
    target_position: Position | None = None,
) -> ShieldCandidateEvaluation:
    """Evaluate one MOVE or WAIT at start/midpoint/end for every human.

    MOVE duration is actual-pose-to-target distance divided by its execution
    speed. WAIT always uses the nominal space-time action duration and keeps
    the actual continuous robot pose fixed.
    """
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")
    if robot_speed <= 0.0:
        raise ValueError("robot_speed must be positive")
    if collision_distance <= 0.0:
        raise ValueError("collision_distance must be positive")
    if move_speed_scale <= 0.0:
        raise ValueError("move_speed_scale must be positive")
    if action not in LOCAL_ACTION_ORDER:
        raise ValueError("unsupported local action")

    action_offset = dict(LOCAL_ACTIONS)[action]
    resolved_cell = target_cell or (
        mapped_start[0] + action_offset[0],
        mapped_start[1] + action_offset[1],
    )
    if action == "WAIT":
        resolved_position = actual_robot_position
        distance = 0.0
        execution_speed = 0.0
        duration = grid_scale / robot_speed
    else:
        resolved_position = (
            target_position
            if target_position is not None
            else grid_to_world(resolved_cell, grid_scale)
        )
        distance = hypot(
            resolved_position[0] - actual_robot_position[0],
            resolved_position[1] - actual_robot_position[1],
        )
        execution_speed = robot_speed * move_speed_scale
        duration = distance / execution_speed

    pedestrians = tuple(pedestrian_states)
    separations = tuple(
        _continuous_motion_separations(
            actual_robot_position,
            resolved_position,
            duration,
            pedestrian_position=position,
            pedestrian_velocity=velocity,
            pedestrian_target=target,
            start_time=0.0,
        )
        for position, velocity, target in pedestrians
    )
    minimum_separation = min(
        (separation for samples in separations for separation in samples),
        default=inf,
    )
    starts_in_collision = any(
        samples[0] <= collision_distance for samples in separations
    )
    unsafe_human_indices = tuple(
        index
        for index, samples in enumerate(separations)
        if min(samples) <= collision_distance
    )
    static_valid = grid_map.is_inside(resolved_cell) and grid_map.is_free(
        resolved_cell
    )

    if not static_valid:
        safe = False
        rejection_reason: ShieldRejectionReason | None = (
            "outside_map"
            if not grid_map.is_inside(resolved_cell)
            else "obstacle"
        )
        collision_egress = False
    elif starts_in_collision:
        collision_egress = is_multi_collision_egress_motion_safe(
            actual_robot_position,
            resolved_position,
            (
                (position, samples)
                for (position, _, _), samples in zip(
                    pedestrians,
                    separations,
                )
            ),
            collision_distance,
        )
        safe = collision_egress
        rejection_reason = None if safe else "non_improving_egress"
    else:
        collision_egress = False
        safe = not unsafe_human_indices
        rejection_reason = None if safe else "predicted_collision"

    return ShieldCandidateEvaluation(
        action=action,
        target_cell=resolved_cell,
        target_position=resolved_position,
        speed_scale=(0.0 if action == "WAIT" else move_speed_scale),
        execution_speed=execution_speed,
        distance=distance,
        duration=duration,
        static_valid=static_valid,
        per_human_separations=separations,
        unsafe_human_indices=unsafe_human_indices,
        minimum_predicted_separation=minimum_separation,
        starts_in_collision=starts_in_collision,
        collision_egress=collision_egress,
        safe=safe,
        rejection_reason=rejection_reason,
    )


def evaluate_local_candidates(
    grid_map: GridMap,
    actual_robot_position: Position,
    mapped_start: Coordinate,
    pedestrian_states: Iterable[PedestrianPredictionState],
    *,
    grid_scale: float,
    robot_speed: float,
    collision_distance: float,
) -> tuple[ShieldCandidateEvaluation, ...]:
    """Evaluate the frozen UP/RIGHT/DOWN/LEFT/WAIT alternate set."""
    pedestrians = tuple(pedestrian_states)
    starts_in_collision = any(
        hypot(
            actual_robot_position[0] - position[0],
            actual_robot_position[1] - position[1],
        )
        <= collision_distance
        for position, _, _ in pedestrians
    )
    move_speed_scale = ESCAPE_SPEED_SCALE if starts_in_collision else 1.0
    evaluations = tuple(
        evaluate_local_action(
            grid_map,
            actual_robot_position,
            mapped_start,
            action,
            pedestrians,
            grid_scale=grid_scale,
            robot_speed=robot_speed,
            collision_distance=collision_distance,
            move_speed_scale=move_speed_scale,
        )
        for action, _ in LOCAL_ACTIONS
    )

    if starts_in_collision and any(
        candidate.safe
        and candidate.per_human_separations
        and all(
            samples[-1] > collision_distance
            for samples in candidate.per_human_separations
        )
        for candidate in evaluations
    ):
        evaluations = tuple(
            replace(
                candidate,
                safe=False,
                rejection_reason="egress_does_not_exit_collision",
            )
            if (
                candidate.safe
                and any(
                    samples[-1] <= collision_distance
                    for samples in candidate.per_human_separations
                )
            )
            else candidate
            for candidate in evaluations
        )
    return evaluations


def select_safe_local_action(
    evaluations: Iterable[ShieldCandidateEvaluation],
    *,
    progress_target: Position,
) -> ShieldCandidateEvaluation | None:
    """Select a safe alternate using the frozen lexicographic policy."""
    safe_candidates = [candidate for candidate in evaluations if candidate.safe]
    if not safe_candidates:
        return None
    return min(
        safe_candidates,
        key=lambda candidate: (
            -candidate.minimum_predicted_separation,
            hypot(
                candidate.target_position[0] - progress_target[0],
                candidate.target_position[1] - progress_target[1],
            ),
            candidate.distance,
            LOCAL_ACTION_ORDER[candidate.action],
            candidate.target_cell[0],
            candidate.target_cell[1],
        ),
    )
