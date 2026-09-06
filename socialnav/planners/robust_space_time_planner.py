"""Continuous-start and safe-egress space-time planning for the robust method."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from math import floor, hypot, inf
from typing import Literal

from socialnav.env.demo_map import grid_to_world
from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.metrics import Position
from socialnav.planners.directional_avoidance import (
    DIRECTION_DOT_TOLERANCE,
    is_separation_increasing,
)
from socialnav.planners.pedestrian_prediction import (
    PedestrianPredictionState,
    predict_pedestrian_position_at_time,
)
from socialnav.planners.social_cost import compute_social_cost
from socialnav.planners.space_time_planner import (
    SpaceTimeAction,
    SpaceTimeActionSafety,
    SpaceTimeFailureReason,
    SpaceTimePlan,
    SpaceTimePlanningResult,
    SpaceTimeSearchStatistics,
    State,
    _collect_pedestrian_states,
)

RobustPlanningFailureReason = Literal[
    "mapping_bridge_failure",
    "no_safe_egress",
    "ordinary_spacetime_planning_failure",
    "other",
]
BridgeRejectionReason = Literal[
    "static_invalid",
    "predicted_collision",
    "non_improving_egress",
    "egress_does_not_exit_collision",
]

EGRESS_SEPARATION_TOLERANCE = DIRECTION_DOT_TOLERANCE

_ACTIONS: tuple[tuple[SpaceTimeAction, Coordinate], ...] = (
    ("UP", (0, -1)),
    ("RIGHT", (1, 0)),
    ("DOWN", (0, 1)),
    ("LEFT", (-1, 0)),
    ("WAIT", (0, 0)),
)
_BRIDGE_OFFSETS: tuple[Coordinate, ...] = (
    (0, 0),
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


@dataclass(frozen=True)
class BridgeCandidateEvaluation:
    """Safety evidence for one deterministic continuous-start candidate."""

    target_cell: Coordinate
    target_position: Position
    static_valid: bool
    distance: float
    duration: float
    predicted_separation_start: float
    predicted_separation_midpoint: float
    predicted_separation_end: float
    minimum_predicted_separation: float
    collision_egress: bool
    safe: bool
    rejection_reason: BridgeRejectionReason | None


@dataclass(frozen=True)
class ContinuousStartBridge:
    """A physical segment from the robot pose to a selected grid centre."""

    start_position: Position
    target_cell: Coordinate
    target_position: Position
    distance: float
    duration: float
    minimum_predicted_separation: float
    collision_egress: bool


@dataclass(frozen=True)
class RobustSpaceTimePlan:
    """A continuous bridge followed by an ordinary time-expanded grid plan."""

    bridge: ContinuousStartBridge
    grid_plan: SpaceTimePlan

    @property
    def estimated_duration(self) -> float:
        return self.bridge.duration + self.grid_plan.estimated_duration


@dataclass(frozen=True)
class RobustSpaceTimePlanningResult:
    """Robust plan plus bridge and downstream search evidence."""

    plan: RobustSpaceTimePlan | None
    failure_reason: RobustPlanningFailureReason | None
    bridge_candidates: tuple[BridgeCandidateEvaluation, ...]
    bridge: ContinuousStartBridge | None
    grid_planning_result: SpaceTimePlanningResult | None


def is_multi_collision_egress_motion_safe(
    start_position: Position,
    end_position: Position,
    humans: Iterable[tuple[Position, tuple[float, float, float]]],
    collision_distance: float,
    *,
    tolerance: float = EGRESS_SEPARATION_TOLERANCE,
) -> bool:
    """Require monotonic escape from every colliding human and safety from
    every currently-safe human.

    Each ``humans`` entry is ``(pedestrian_at_start, separations)``.  A
    currently-colliding pedestrian must pass the existing single-human
    strictly-increasing egress test; a currently-safe pedestrian must keep
    every sampled separation above the collision distance.  Never allow an
    escape from pedestrian A that creates a collision with pedestrian B.
    """
    if collision_distance <= 0.0:
        raise ValueError("collision_distance must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    for pedestrian_at_start, separations in humans:
        start_separation = separations[0]
        if start_separation <= collision_distance:
            if not is_collision_egress_motion_safe(
                start_position,
                end_position,
                pedestrian_at_start,
                separations,
                collision_distance,
                tolerance=tolerance,
            ):
                return False
        elif not all(
            separation > collision_distance for separation in separations
        ):
            return False
    return True


def build_continuous_start_transitions(
    grid_map: GridMap,
    actual_start_position: Position,
    mapped_start: Coordinate,
    *,
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    grid_scale: float,
    robot_speed: float,
    collision_distance: float,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> tuple[BridgeCandidateEvaluation, ...]:
    """Evaluate the mapped cell and its four neighbors without broad search.

    Every candidate is checked against every pedestrian; the reported
    separations are the minimum across all pedestrians at each of the three
    time-aligned samples.
    """
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")
    if robot_speed <= 0.0:
        raise ValueError("robot_speed must be positive")
    if collision_distance <= 0.0:
        raise ValueError("collision_distance must be positive")

    additional = tuple(additional_pedestrians)
    pedestrians = _collect_pedestrian_states(
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        additional,
    )
    candidate_cells = tuple(
        sorted(
            {
                (mapped_start[0] + delta_x, mapped_start[1] + delta_y)
                for delta_x, delta_y in _BRIDGE_OFFSETS
            }
        )
    )
    starts_unsafe = any(
        hypot(
            actual_start_position[0] - position[0],
            actual_start_position[1] - position[1],
        )
        <= collision_distance
        for position, _, _ in pedestrians
    )
    evaluations = tuple(
        _evaluate_bridge_candidate(
            grid_map,
            actual_start_position,
            candidate,
            pedestrian_position=pedestrian_position,
            pedestrian_velocity=pedestrian_velocity,
            pedestrian_target=pedestrian_target,
            grid_scale=grid_scale,
            robot_speed=robot_speed,
            collision_distance=collision_distance,
            collision_egress=starts_unsafe,
            additional_pedestrians=additional,
        )
        for candidate in candidate_cells
    )

    if starts_unsafe and any(
        evaluation.safe
        and evaluation.predicted_separation_end > collision_distance
        for evaluation in evaluations
    ):
        evaluations = tuple(
            (
                replace(
                    evaluation,
                    safe=False,
                    rejection_reason="egress_does_not_exit_collision",
                )
                if (
                    evaluation.safe
                    and evaluation.predicted_separation_end
                    <= collision_distance
                )
                else evaluation
            )
            for evaluation in evaluations
        )
    return evaluations


def select_continuous_start_bridge(
    evaluations: tuple[BridgeCandidateEvaluation, ...],
    actual_start_position: Position,
) -> ContinuousStartBridge | None:
    """Select a safe bridge by the frozen deterministic tie-breaking rule."""
    safe_candidates = [
        evaluation for evaluation in evaluations if evaluation.safe
    ]
    if not safe_candidates:
        return None
    selected = min(
        safe_candidates,
        key=lambda candidate: (
            candidate.distance,
            -candidate.minimum_predicted_separation,
            candidate.target_cell[0],
            candidate.target_cell[1],
        ),
    )
    return ContinuousStartBridge(
        start_position=actual_start_position,
        target_cell=selected.target_cell,
        target_position=selected.target_position,
        distance=selected.distance,
        duration=selected.duration,
        minimum_predicted_separation=selected.minimum_predicted_separation,
        collision_egress=selected.collision_egress,
    )


def interpolate_bridge_position(
    bridge: ContinuousStartBridge,
    fraction: float,
) -> Position:
    """Return a point on a bridge without changing the physical robot state."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    return (
        bridge.start_position[0]
        + (bridge.target_position[0] - bridge.start_position[0]) * fraction,
        bridge.start_position[1]
        + (bridge.target_position[1] - bridge.start_position[1]) * fraction,
    )


def is_collision_egress_motion_safe(
    start_position: Position,
    end_position: Position,
    pedestrian_at_start: Position,
    separations: tuple[float, float, float],
    collision_distance: float,
    *,
    tolerance: float = EGRESS_SEPARATION_TOLERANCE,
) -> bool:
    """Allow only strictly separation-increasing motion from an unsafe start."""
    if collision_distance <= 0.0:
        raise ValueError("collision_distance must be positive")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    start_separation, midpoint_separation, end_separation = separations
    if start_separation > collision_distance:
        return False
    intended_motion = (
        end_position[0] - start_position[0],
        end_position[1] - start_position[1],
    )
    if not is_separation_increasing(
        start_position,
        pedestrian_at_start,
        intended_motion,
        tolerance=tolerance,
    ):
        return False
    return (
        midpoint_separation > start_separation + tolerance
        and end_separation > midpoint_separation + tolerance
    )


def robust_space_time_social_astar(
    grid_map: GridMap,
    actual_start_position: Position,
    mapped_start: Coordinate,
    goal: Coordinate,
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    social_distance: float,
    social_weight: float,
    collision_distance: float,
    grid_scale: float,
    robot_speed: float,
    max_time_seconds: float,
    *,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> RobustSpaceTimePlanningResult:
    """Bridge from the continuous pose, then search with its time offset.

    With no additional pedestrians this reproduces the original
    single-human robust planner.
    """
    if max_time_seconds < 0.0:
        raise ValueError("max_time_seconds must be non-negative")

    additional = tuple(additional_pedestrians)
    pedestrians = _collect_pedestrian_states(
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        additional,
    )
    evaluations = build_continuous_start_transitions(
        grid_map,
        actual_start_position,
        mapped_start,
        pedestrian_position=pedestrian_position,
        pedestrian_velocity=pedestrian_velocity,
        pedestrian_target=pedestrian_target,
        grid_scale=grid_scale,
        robot_speed=robot_speed,
        collision_distance=collision_distance,
        additional_pedestrians=additional,
    )
    bridge = select_continuous_start_bridge(
        evaluations,
        actual_start_position,
    )
    starts_unsafe = any(
        hypot(
            actual_start_position[0] - position[0],
            actual_start_position[1] - position[1],
        )
        <= collision_distance
        for position, _, _ in pedestrians
    )
    if bridge is None or bridge.duration > max_time_seconds:
        return RobustSpaceTimePlanningResult(
            plan=None,
            failure_reason=(
                "no_safe_egress"
                if starts_unsafe
                else "mapping_bridge_failure"
            ),
            bridge_candidates=evaluations,
            bridge=None,
            grid_planning_result=None,
        )

    grid_result = _search_from_bridge(
        grid_map,
        bridge,
        goal,
        pedestrian_position=pedestrian_position,
        pedestrian_velocity=pedestrian_velocity,
        pedestrian_target=pedestrian_target,
        social_distance=social_distance,
        social_weight=social_weight,
        collision_distance=collision_distance,
        grid_scale=grid_scale,
        robot_speed=robot_speed,
        max_time_seconds=max_time_seconds,
        additional_pedestrians=additional,
    )
    if grid_result.plan is None:
        first_actions = grid_result.first_action_safety
        no_egress = (
            starts_unsafe
            and not any(
                detail.rejection_reason is None for detail in first_actions
            )
        )
        return RobustSpaceTimePlanningResult(
            plan=None,
            failure_reason=(
                "no_safe_egress"
                if no_egress
                else "ordinary_spacetime_planning_failure"
            ),
            bridge_candidates=evaluations,
            bridge=bridge,
            grid_planning_result=grid_result,
        )

    return RobustSpaceTimePlanningResult(
        plan=RobustSpaceTimePlan(bridge=bridge, grid_plan=grid_result.plan),
        failure_reason=None,
        bridge_candidates=evaluations,
        bridge=bridge,
        grid_planning_result=grid_result,
    )
def _evaluate_bridge_candidate(
    grid_map: GridMap,
    actual_start_position: Position,
    target_cell: Coordinate,
    *,
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    grid_scale: float,
    robot_speed: float,
    collision_distance: float,
    collision_egress: bool,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> BridgeCandidateEvaluation:
    target_position = grid_to_world(target_cell, grid_scale)
    distance = hypot(
        target_position[0] - actual_start_position[0],
        target_position[1] - actual_start_position[1],
    )
    duration = distance / robot_speed
    separations = _continuous_motion_separations(
        actual_start_position,
        target_position,
        duration,
        pedestrian_position=pedestrian_position,
        pedestrian_velocity=pedestrian_velocity,
        pedestrian_target=pedestrian_target,
        start_time=0.0,
        additional_pedestrians=additional_pedestrians,
    )
    static_valid = grid_map.is_inside(target_cell) and grid_map.is_free(
        target_cell
    )
    if not static_valid:
        safe = False
        rejection_reason: BridgeRejectionReason | None = "static_invalid"
    elif collision_egress:
        pedestrians = _collect_pedestrian_states(
            pedestrian_position,
            pedestrian_velocity,
            pedestrian_target,
            additional_pedestrians,
        )
        humans = [
            (
                position,
                _continuous_motion_separations(
                    actual_start_position,
                    target_position,
                    duration,
                    pedestrian_position=position,
                    pedestrian_velocity=velocity,
                    pedestrian_target=target,
                    start_time=0.0,
                ),
            )
            for position, velocity, target in pedestrians
        ]
        safe = is_multi_collision_egress_motion_safe(
            actual_start_position,
            target_position,
            humans,
            collision_distance,
        )
        rejection_reason = None if safe else "non_improving_egress"
    else:
        safe = all(
            separation > collision_distance for separation in separations
        )
        rejection_reason = None if safe else "predicted_collision"

    return BridgeCandidateEvaluation(
        target_cell=target_cell,
        target_position=target_position,
        static_valid=static_valid,
        distance=distance,
        duration=duration,
        predicted_separation_start=separations[0],
        predicted_separation_midpoint=separations[1],
        predicted_separation_end=separations[2],
        minimum_predicted_separation=min(separations),
        collision_egress=collision_egress,
        safe=safe,
        rejection_reason=rejection_reason,
    )


def _continuous_motion_separations(
    start_position: Position,
    end_position: Position,
    duration: float,
    *,
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    start_time: float,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> tuple[float, float, float]:
    pedestrians = _collect_pedestrian_states(
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        additional_pedestrians,
    )
    separations_by_fraction: list[list[float]] = [[], [], []]
    for position, velocity, target in pedestrians:
        for fraction_index, fraction in enumerate((0.0, 0.5, 1.0)):
            robot_at_time = (
                start_position[0]
                + (end_position[0] - start_position[0]) * fraction,
                start_position[1]
                + (end_position[1] - start_position[1]) * fraction,
            )
            pedestrian_at_time = predict_pedestrian_position_at_time(
                position,
                velocity,
                start_time + duration * fraction,
                target=target,
            )
            separations_by_fraction[fraction_index].append(
                hypot(
                    robot_at_time[0] - pedestrian_at_time[0],
                    robot_at_time[1] - pedestrian_at_time[1],
                )
            )
    return (
        min(separations_by_fraction[0], default=inf),
        min(separations_by_fraction[1], default=inf),
        min(separations_by_fraction[2], default=inf),
    )


def _inspect_grid_actions(
    grid_map: GridMap,
    start: Coordinate,
    start_time_index: int,
    *,
    time_offset: float,
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    move_duration: float,
    grid_scale: float,
    collision_distance: float,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> tuple[SpaceTimeActionSafety, ...]:
    start_position = grid_to_world(start, grid_scale)
    pedestrians = _collect_pedestrian_states(
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        additional_pedestrians,
    )
    action_data = []
    for action, (delta_x, delta_y) in _ACTIONS:
        destination = (start[0] + delta_x, start[1] + delta_y)
        destination_position = grid_to_world(destination, grid_scale)
        inside_map = grid_map.is_inside(destination)
        free = inside_map and grid_map.is_free(destination)
        start_time = time_offset + start_time_index * move_duration
        separations = _continuous_motion_separations(
            start_position,
            destination_position,
            move_duration,
            pedestrian_position=pedestrian_position,
            pedestrian_velocity=pedestrian_velocity,
            pedestrian_target=pedestrian_target,
            start_time=start_time,
            additional_pedestrians=additional_pedestrians,
        )
        humans = [
            (
                predict_pedestrian_position_at_time(
                    position,
                    velocity,
                    start_time,
                    target=target,
                ),
                _continuous_motion_separations(
                    start_position,
                    destination_position,
                    move_duration,
                    pedestrian_position=position,
                    pedestrian_velocity=velocity,
                    pedestrian_target=target,
                    start_time=start_time,
                ),
            )
            for position, velocity, target in pedestrians
        ]
        ordinary_safe = all(
            separation > collision_distance for separation in separations
        )
        egress_safe = is_multi_collision_egress_motion_safe(
            start_position,
            destination_position,
            humans,
            collision_distance,
        )
        temporal_safe = ordinary_safe or egress_safe
        if not inside_map:
            rejection_reason = "outside_map"
        elif not free:
            rejection_reason = "obstacle"
        elif not temporal_safe:
            rejection_reason = "predicted_collision"
        else:
            rejection_reason = None
        action_data.append(
            (
                SpaceTimeActionSafety(
                    action=action,
                    destination=destination,
                    inside_map=inside_map,
                    free=free,
                    predicted_separation_start=separations[0],
                    predicted_separation_midpoint=separations[1],
                    predicted_separation_end=separations[2],
                    minimum_predicted_separation=min(separations),
                    rejected_by_collision_constraint=not temporal_safe,
                    rejection_reason=rejection_reason,
                ),
                egress_safe,
            )
        )

    if any(
        egress
        and detail.rejection_reason is None
        and detail.predicted_separation_end > collision_distance
        for detail, egress in action_data
    ):
        action_data = [
            (
                replace(
                    detail,
                    rejected_by_collision_constraint=True,
                    rejection_reason="predicted_collision",
                ),
                egress,
            )
            if (
                egress
                and detail.rejection_reason is None
                and detail.predicted_separation_end <= collision_distance
            )
            else (detail, egress)
            for detail, egress in action_data
        ]
    return tuple(detail for detail, _ in action_data)
def _search_from_bridge(
    grid_map: GridMap,
    bridge: ContinuousStartBridge,
    goal: Coordinate,
    *,
    pedestrian_position: Position | None,
    pedestrian_velocity: Position,
    pedestrian_target: Position | None,
    social_distance: float,
    social_weight: float,
    collision_distance: float,
    grid_scale: float,
    robot_speed: float,
    max_time_seconds: float,
    additional_pedestrians: Iterable[PedestrianPredictionState] = (),
) -> SpaceTimePlanningResult:
    if social_distance <= 0.0:
        raise ValueError("social_distance must be positive")
    if social_weight < 0.0:
        raise ValueError("social_weight must be non-negative")

    additional = tuple(additional_pedestrians)
    pedestrians = _collect_pedestrian_states(
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        additional,
    )
    start = bridge.target_cell
    move_duration = grid_scale / robot_speed
    max_time_index = floor(
        (max_time_seconds - bridge.duration) / move_duration
    )
    empty_statistics = SpaceTimeSearchStatistics(
        expanded_states=0,
        generated_states=0,
        maximum_time_index_reached=0,
        planning_horizon_reached=False,
        open_set_exhausted=False,
        goal_reached=False,
        returned_path_length=None,
        planned_wait_count=0,
    )
    if not grid_map.is_inside(start) or not grid_map.is_free(start):
        return SpaceTimePlanningResult(
            None, "invalid_start", (), empty_statistics
        )
    if not grid_map.is_inside(goal) or not grid_map.is_free(goal):
        return SpaceTimePlanningResult(
            None, "invalid_goal", (), empty_statistics
        )

    first_action_safety = _inspect_grid_actions(
        grid_map,
        start,
        0,
        time_offset=bridge.duration,
        pedestrian_position=pedestrian_position,
        pedestrian_velocity=pedestrian_velocity,
        pedestrian_target=pedestrian_target,
        move_duration=move_duration,
        grid_scale=grid_scale,
        collision_distance=collision_distance,
        additional_pedestrians=additional,
    )
    start_state: State = (start[0], start[1], 0)
    if start == goal:
        plan = SpaceTimePlan(
            spatial_path=(start,),
            timed_states=(start_state,),
            actions=(),
            planned_wait_actions=0,
            planned_move_actions=0,
            move_duration=move_duration,
            estimated_duration=0.0,
        )
        return SpaceTimePlanningResult(
            plan,
            None,
            first_action_safety,
            SpaceTimeSearchStatistics(
                expanded_states=1,
                generated_states=1,
                maximum_time_index_reached=0,
                planning_horizon_reached=max_time_index == 0,
                open_set_exhausted=False,
                goal_reached=True,
                returned_path_length=0,
                planned_wait_count=0,
            ),
        )

    tie_breaker = count()
    frontier: list[tuple[float, int, float, State]] = []
    heappush(
        frontier,
        (float(_manhattan(start, goal)), next(tie_breaker), 0.0, start_state),
    )
    cost_so_far = {start_state: 0.0}
    came_from: dict[State, tuple[State, SpaceTimeAction]] = {}
    expanded_states = 0
    generated_states = 1
    maximum_time_index_reached = 0

    while frontier:
        _, _, current_cost, current = heappop(frontier)
        if current_cost != cost_so_far[current]:
            continue
        expanded_states += 1
        maximum_time_index_reached = max(
            maximum_time_index_reached,
            current[2],
        )
        current_coordinate = (current[0], current[1])
        if current_coordinate == goal:
            plan = _reconstruct_plan(
                came_from,
                start_state,
                current,
                move_duration,
            )
            return SpaceTimePlanningResult(
                plan,
                None,
                first_action_safety,
                SpaceTimeSearchStatistics(
                    expanded_states=expanded_states,
                    generated_states=generated_states,
                    maximum_time_index_reached=maximum_time_index_reached,
                    planning_horizon_reached=(
                        maximum_time_index_reached >= max_time_index
                    ),
                    open_set_exhausted=False,
                    goal_reached=True,
                    returned_path_length=len(plan.actions),
                    planned_wait_count=plan.planned_wait_actions,
                ),
            )
        if current[2] >= max_time_index:
            continue

        action_safety = _inspect_grid_actions(
            grid_map,
            current_coordinate,
            current[2],
            time_offset=bridge.duration,
            pedestrian_position=pedestrian_position,
            pedestrian_velocity=pedestrian_velocity,
            pedestrian_target=pedestrian_target,
            move_duration=move_duration,
            grid_scale=grid_scale,
            collision_distance=collision_distance,
            additional_pedestrians=additional,
        )
        for detail in action_safety:
            if detail.rejection_reason is not None:
                continue
            neighbor = detail.destination
            next_state: State = (neighbor[0], neighbor[1], current[2] + 1)
            arrival_time = bridge.duration + next_state[2] * move_duration
            social_penalty = sum(
                compute_social_cost(
                    grid_to_world(neighbor, grid_scale),
                    predict_pedestrian_position_at_time(
                        position,
                        velocity,
                        arrival_time,
                        target=target,
                    ),
                    social_distance,
                    social_weight,
                )
                for position, velocity, target in pedestrians
            )
            new_cost = current_cost + 1.0 + social_penalty
            if new_cost >= cost_so_far.get(next_state, float("inf")):
                continue
            cost_so_far[next_state] = new_cost
            came_from[next_state] = (current, detail.action)
            generated_states += 1
            maximum_time_index_reached = max(
                maximum_time_index_reached,
                next_state[2],
            )
            priority = new_cost + _manhattan(neighbor, goal)
            heappush(
                frontier,
                (priority, next(tie_breaker), new_cost, next_state),
            )

    planning_horizon_reached = maximum_time_index_reached >= max_time_index
    start_separation = first_action_safety[0].predicted_separation_start
    has_safe_first_action = any(
        detail.rejection_reason is None for detail in first_action_safety
    )
    if start_separation <= collision_distance:
        failure_reason: SpaceTimeFailureReason = (
            "start_in_predicted_collision"
        )
    elif not has_safe_first_action:
        failure_reason = "no_safe_first_action"
    elif not _is_static_goal_reachable(grid_map, start, goal):
        failure_reason = "goal_unreachable_static"
    elif planning_horizon_reached:
        failure_reason = "time_horizon_exhausted"
    elif expanded_states > 0:
        failure_reason = "search_exhausted"
    else:
        failure_reason = "other"
    return SpaceTimePlanningResult(
        None,
        failure_reason,
        first_action_safety,
        SpaceTimeSearchStatistics(
            expanded_states=expanded_states,
            generated_states=generated_states,
            maximum_time_index_reached=maximum_time_index_reached,
            planning_horizon_reached=planning_horizon_reached,
            open_set_exhausted=True,
            goal_reached=False,
            returned_path_length=None,
            planned_wait_count=0,
        ),
    )


def _is_static_goal_reachable(
    grid_map: GridMap,
    start: Coordinate,
    goal: Coordinate,
) -> bool:
    frontier = [start]
    visited = {start}
    while frontier:
        current = frontier.pop()
        if current == goal:
            return True
        for _, (delta_x, delta_y) in _ACTIONS[:-1]:
            neighbor = (current[0] + delta_x, current[1] + delta_y)
            if (
                neighbor not in visited
                and grid_map.is_inside(neighbor)
                and grid_map.is_free(neighbor)
            ):
                visited.add(neighbor)
                frontier.append(neighbor)
    return False


def _manhattan(first: Coordinate, second: Coordinate) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _reconstruct_plan(
    came_from: dict[State, tuple[State, SpaceTimeAction]],
    start: State,
    goal: State,
    move_duration: float,
) -> SpaceTimePlan:
    states = [goal]
    actions: list[SpaceTimeAction] = []
    current = goal
    while current != start:
        previous, action = came_from[current]
        states.append(previous)
        actions.append(action)
        current = previous
    states.reverse()
    actions.reverse()
    action_tuple = tuple(actions)
    timed_states = tuple(states)
    return SpaceTimePlan(
        spatial_path=tuple((state[0], state[1]) for state in timed_states),
        timed_states=timed_states,
        actions=action_tuple,
        planned_wait_actions=action_tuple.count("WAIT"),
        planned_move_actions=len(action_tuple) - action_tuple.count("WAIT"),
        move_duration=move_duration,
        estimated_duration=len(action_tuple) * move_duration,
    )