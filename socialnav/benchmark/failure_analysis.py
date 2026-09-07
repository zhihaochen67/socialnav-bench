"""Deterministic Phase 7C failure analysis for robust space-time episodes.

Every rule in this module consumes only recorded benchmark evidence
(EpisodeResult, EpisodeTrace, Scenario, and optionally recorded
trajectories from the evidence probe).  Nothing here changes planner
behavior, scenario generation, or the episode timeout.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot
from statistics import mean, median

from socialnav.benchmark.diagnostics import EpisodeTrace
from socialnav.benchmark.replanning import world_to_nearest_free_cell
from socialnav.benchmark.scenario import Scenario, build_scenario_grid
from socialnav.env.demo_map import SOCIAL_DISTANCE, grid_to_world
from socialnav.env.grid_map import GridMap
from socialnav.env.pedestrian import compute_pedestrian_velocity
from socialnav.env.world import (
    HUMAN_COLLISION_DISTANCE,
    ROBOT_SPEED,
    SIMULATION_STEP,
    SLOW_DISTANCE,
    STOP_DISTANCE,
)
from socialnav.evaluation import EpisodeResult
from socialnav.metrics import Position
from socialnav.planners.astar import astar
from socialnav.planners.directional_avoidance import (
    DIRECTION_DOT_TOLERANCE,
    is_separation_increasing,
)
from socialnav.planners.pedestrian_prediction import (
    PREDICTION_EPSILON,
    predict_pedestrian_position_at_time,
)
from socialnav.planners.robust_space_time_planner import (
    build_continuous_start_transitions,
    is_multi_collision_egress_motion_safe,
)

BRIDGE_FAILURE_REASONS = ("mapping_bridge_failure", "no_safe_egress")

FAILURE_CATEGORIES: tuple[str, ...] = (
    "multi_human_egress_conflict",
    "human_collision",
    "static_goal_unreachable",
    "no_safe_first_action",
    "spacetime_search_exhausted",
    "time_horizon_exhausted",
    "repeated_replan_failure",
    "no_safe_replan_bridge",
    "no_safe_initial_bridge",
    "multi_pedestrian_corridor_blockage",
    "pedestrian_terminal_blockage",
    "reactive_execution_deadlock",
    "progress_stall_timeout",
    "other",
)

PLANNING_EXECUTION_CLASSES: tuple[str, ...] = (
    "planner_no_valid_plan",
    "execution_progressed_then_collided",
    "repeated_replan_failure_unsafe_state",
    "planner_ok_execution_deadlock",
    "apparently_infeasible_horizon",
)

ACTIONS: tuple[str, ...] = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT")
_ACTION_OFFSETS: dict[str, tuple[int, int]] = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
}

_REQUIRED_GRID_FAILURE_REASONS = {
    "no_safe_first_action",
    "search_exhausted",
    "time_horizon_exhausted",
    "goal_unreachable_static",
}

# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _point_to_segment_distance(
    point: Position,
    start: Position,
    end: Position,
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    squared_length = delta_x * delta_x + delta_y * delta_y
    if squared_length == 0.0:
        return hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * delta_x
        + (point[1] - start[1]) * delta_y
    ) / squared_length
    projection = min(max(projection, 0.0), 1.0)
    nearest_x = start[0] + projection * delta_x
    nearest_y = start[1] + projection * delta_y
    return hypot(point[0] - nearest_x, point[1] - nearest_y)


def point_to_route_distance(
    point: Position,
    route: tuple[Position, ...],
) -> float:
    """Return the minimum world-space distance from a point to a route."""
    if not route:
        raise ValueError("route must not be empty")
    if len(route) == 1:
        return hypot(point[0] - route[0][0], point[1] - route[0][1])
    return min(
        _point_to_segment_distance(point, start, end)
        for start, end in zip(route, route[1:])
    )


def _orientation(
    first: Position,
    second: Position,
    third: Position,
) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _segments_properly_intersect(
    first: tuple[Position, Position],
    second: tuple[Position, Position],
) -> bool:
    """Strict 2D segment intersection with four orientation tests."""
    (a_x, a_y), (b_x, b_y) = first
    (c_x, c_y), (d_x, d_y) = second
    orientation_abc = _orientation((a_x, a_y), (b_x, b_y), (c_x, c_y))
    orientation_abd = _orientation((a_x, a_y), (b_x, b_y), (d_x, d_y))
    orientation_cda = _orientation((c_x, c_y), (d_x, d_y), (a_x, a_y))
    orientation_cdb = _orientation((c_x, c_y), (d_x, d_y), (b_x, b_y))
    return (
        orientation_abc * orientation_abd < 0.0
        and orientation_cda * orientation_cdb < 0.0
    )


def segment_intersects_route(
    start: Position,
    end: Position,
    route: tuple[Position, ...],
) -> bool:
    """Return whether a segment properly crosses any route segment."""
    if len(route) < 2:
        return False
    return any(
        _segments_properly_intersect((start, end), route_segment)
        for route_segment in zip(route, route[1:])
    )


# ---------------------------------------------------------------------------
# Terminal-blockage evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerminalBlockageEvidence:
    """Deterministic terminal-pedestrian blockage evidence for one episode."""

    terminal_indices: tuple[int, ...]
    terminal_route_blocker_indices: tuple[int, ...]
    joint_corridor_blockage: bool
    robot_near_blocker: bool
    target_on_or_near_route_count: int


@dataclass(frozen=True)
class PersistentBlockerEvidence:
    """Final-state evidence for one pedestrian in the topology probe.

    ``pedestrian_id`` is the stable zero-based scenario-order identifier used
    throughout the benchmark's multi-human diagnostics.
    """

    pedestrian_id: int
    position: Position
    target: Position
    velocity: Position
    terminal: bool
    stationary: bool
    persistently_collision_relevant: bool
    persistent_terminal_candidate: bool
    unsafe_cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PersistentDynamicBlockageDiagnostic:
    """Static-connectivity evidence refining a raw planner failure reason."""

    planner_failure_reason: str | None
    diagnostic_failure_category: str
    persistent_dynamic_blockage: bool
    persistent_blockage_type: str | None
    robot_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    blocking_pedestrian_ids: tuple[int, ...]
    blocking_cells: tuple[tuple[int, int], ...]
    blocker_count: int
    static_connectivity: bool
    connectivity_with_persistent_blockers: bool
    removing_dynamic_blockers_restores_static_connectivity: bool
    multiple_humans_jointly_form_cut: bool
    pedestrian_evidence: tuple[PersistentBlockerEvidence, ...]


def pedestrian_at_target(
    position: Position,
    target: Position,
    *,
    epsilon: float = PREDICTION_EPSILON,
) -> bool:
    """Return whether a pedestrian is at its target within the fixed epsilon."""
    return (
        hypot(position[0] - target[0], position[1] - target[1]) <= epsilon
    )


def _latest_planner_failure_reason(trace: EpisodeTrace) -> str | None:
    for call in reversed(trace.space_time_planning_calls):
        if call.failure_reason is not None:
            return call.failure_reason
    if trace.robust_planning_failure_reasons:
        return trace.robust_planning_failure_reasons[-1]
    return None


def _is_grid_connected(
    grid_map: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked_cells: set[tuple[int, int]],
) -> bool:
    """Four-connected reachability on the static map plus probe-only cells."""
    if (
        not grid_map.is_free(start)
        or not grid_map.is_free(goal)
        or start in blocked_cells
        or goal in blocked_cells
    ):
        return False

    frontier = [start]
    visited = {start}
    while frontier:
        current = frontier.pop()
        if current == goal:
            return True
        for delta_x, delta_y in _ACTION_OFFSETS.values():
            if (delta_x, delta_y) == (0, 0):
                continue
            neighbor = (current[0] + delta_x, current[1] + delta_y)
            if (
                neighbor not in visited
                and neighbor not in blocked_cells
                and grid_map.is_free(neighbor)
            ):
                visited.add(neighbor)
                frontier.append(neighbor)
    return False


def _blocked_cells_for(
    pedestrian_ids: tuple[int, ...],
    unsafe_by_id: dict[int, set[tuple[int, int]]],
) -> set[tuple[int, int]]:
    return (
        set().union(
            *(unsafe_by_id[pedestrian_id] for pedestrian_id in pedestrian_ids)
        )
        if pedestrian_ids
        else set()
    )


def diagnose_persistent_dynamic_blockage(
    scenario: Scenario,
    trace: EpisodeTrace,
    *,
    planner_failure_reason: str | None = None,
    collision_distance: float = HUMAN_COLLISION_DISTANCE,
) -> PersistentDynamicBlockageDiagnostic:
    """Refine a failed space-time plan using final pedestrian topology.

    The probe is deliberately outside the planner. It maps the robot's final
    pose to the nearest statically free cell, marks every free cell whose
    centre is within the planner's conservative collision radius (``<=``) of
    a pedestrian that is both at its target and stationary, then performs
    four-connected reachability. If the static map is connected but the
    augmented map is not, the smallest deterministic pedestrian subset that
    disconnects robot and goal is reported as the blocking cut set.
    """
    if collision_distance <= 0.0:
        raise ValueError("collision_distance must be positive")

    raw_reason = (
        _latest_planner_failure_reason(trace)
        if planner_failure_reason is None
        else planner_failure_reason
    )
    grid_map = build_scenario_grid(scenario)
    robot_cell = world_to_nearest_free_cell(
        grid_map,
        trace.final_robot_position,
        scenario.grid_scale,
    )
    final_positions = trace.final_pedestrian_positions
    if not final_positions and len(scenario.pedestrians) == 1:
        final_positions = (trace.final_pedestrian_position,)

    pedestrian_rows = []
    unsafe_by_id: dict[int, set[tuple[int, int]]] = {}
    for pedestrian_id, (position, spec) in enumerate(
        zip(final_positions, scenario.pedestrians)
    ):
        velocity = compute_pedestrian_velocity(position, spec.target, spec.speed)
        # Match the runtime Pedestrian semantics: exact target equality makes
        # the pedestrian terminal, and its derived velocity is then exactly
        # zero.  Keep the checks separate so both facts remain observable.
        terminal = position == spec.target
        stationary = velocity == (0.0, 0.0)
        unsafe_cells = set()
        for x in range(grid_map.width):
            for y in range(grid_map.height):
                cell = (x, y)
                if not grid_map.is_free(cell):
                    continue
                cell_position = grid_to_world(cell, scenario.grid_scale)
                if (
                    hypot(
                        cell_position[0] - position[0],
                        cell_position[1] - position[1],
                    )
                    <= collision_distance
                ):
                    unsafe_cells.add(cell)
        collision_relevant = bool(unsafe_cells)
        persistent_candidate = (
            terminal and stationary and collision_relevant
        )
        if persistent_candidate:
            unsafe_by_id[pedestrian_id] = unsafe_cells
        pedestrian_rows.append(
            PersistentBlockerEvidence(
                pedestrian_id=pedestrian_id,
                position=position,
                target=spec.target,
                velocity=velocity,
                terminal=terminal,
                stationary=stationary,
                persistently_collision_relevant=persistent_candidate,
                persistent_terminal_candidate=persistent_candidate,
                unsafe_cells=tuple(sorted(unsafe_cells)),
            )
        )

    static_connectivity = _is_grid_connected(
        grid_map,
        robot_cell,
        scenario.goal,
        set(),
    )
    candidate_ids = tuple(sorted(unsafe_by_id))
    all_blocked_cells = _blocked_cells_for(candidate_ids, unsafe_by_id)
    dynamic_connectivity = _is_grid_connected(
        grid_map,
        robot_cell,
        scenario.goal,
        all_blocked_cells,
    )
    blocking_ids: tuple[int, ...] = ()
    if static_connectivity and not dynamic_connectivity:
        for subset_size in range(1, len(candidate_ids) + 1):
            blocking_ids = next(
                (
                    subset
                    for subset in combinations(candidate_ids, subset_size)
                    if not _is_grid_connected(
                        grid_map,
                        robot_cell,
                        scenario.goal,
                        _blocked_cells_for(subset, unsafe_by_id),
                    )
                ),
                (),
            )
            if blocking_ids:
                break

    blocking_cells = _blocked_cells_for(blocking_ids, unsafe_by_id)
    persistent_blockage = bool(blocking_ids)
    if not static_connectivity:
        diagnostic_category = "static_goal_unreachable"
    elif persistent_blockage:
        diagnostic_category = "persistent_dynamic_blockage"
    elif raw_reason is not None:
        diagnostic_category = raw_reason
    elif trace.robust_episode_failure_reason is not None:
        diagnostic_category = trace.robust_episode_failure_reason
    elif trace.timed_out:
        diagnostic_category = "episode_timeout"
    else:
        diagnostic_category = "other"

    multiple_human_cut = len(blocking_ids) > 1
    return PersistentDynamicBlockageDiagnostic(
        planner_failure_reason=raw_reason,
        diagnostic_failure_category=diagnostic_category,
        persistent_dynamic_blockage=persistent_blockage,
        persistent_blockage_type=(
            "multi_human_corridor_cut_set_blockage"
            if multiple_human_cut
            else "terminal_pedestrian_blockage"
            if persistent_blockage
            else None
        ),
        robot_cell=robot_cell,
        goal_cell=scenario.goal,
        blocking_pedestrian_ids=blocking_ids,
        blocking_cells=tuple(sorted(blocking_cells)),
        blocker_count=len(blocking_ids),
        static_connectivity=static_connectivity,
        connectivity_with_persistent_blockers=dynamic_connectivity,
        removing_dynamic_blockers_restores_static_connectivity=(
            persistent_blockage and static_connectivity
        ),
        multiple_humans_jointly_form_cut=multiple_human_cut,
        pedestrian_evidence=tuple(pedestrian_rows),
    )


def compute_terminal_blockage_evidence(
    scenario: Scenario,
    trace: EpisodeTrace,
    *,
    route_stop_distance: float = STOP_DISTANCE,
    cluster_distance: float = SLOW_DISTANCE,
) -> TerminalBlockageEvidence:
    """Identify pedestrians that ended the episode at their targets on or
    near the robot's planned route, and whether several of them jointly
    occupy a corridor around the route.

    Thresholds are the existing STOP_DISTANCE / SLOW_DISTANCE constants,
    not benchmark-specific tuned values.
    """
    final_positions = trace.final_pedestrian_positions
    targets = scenario.pedestrians
    terminal_indices = tuple(
        index
        for index, (position, spec) in enumerate(
            zip(final_positions, targets)
        )
        if pedestrian_at_target(position, spec.target)
    )
    route = trace.planned_path
    terminal_route_blocker_indices = tuple(
        index
        for index in terminal_indices
        if point_to_route_distance(
            targets[index].target,
            route,
        )
        <= route_stop_distance
    )
    blocker_targets = [
        targets[index].target for index in terminal_route_blocker_indices
    ]
    joint_corridor_blockage = (
        len(blocker_targets) >= 2
        and all(
            hypot(
                first[0] - second[0],
                first[1] - second[1],
            )
            <= cluster_distance
            for first in blocker_targets
            for second in blocker_targets
            if first != second
        )
    )
    robot_near_blocker = any(
        hypot(
            trace.final_robot_position[0] - target[0],
            trace.final_robot_position[1] - target[1],
        )
        <= SLOW_DISTANCE
        for target in blocker_targets
    )
    return TerminalBlockageEvidence(
        terminal_indices=terminal_indices,
        terminal_route_blocker_indices=terminal_route_blocker_indices,
        joint_corridor_blockage=joint_corridor_blockage,
        robot_near_blocker=robot_near_blocker,
        target_on_or_near_route_count=sum(
            point_to_route_distance(spec.target, route)
            <= STOP_DISTANCE
            for spec in targets
        ),
    )


# ---------------------------------------------------------------------------
# Primary failure taxonomy
# ---------------------------------------------------------------------------

def _any_planning_call_failure_reason(
    trace: EpisodeTrace,
    reason: str,
) -> bool:
    return any(
        call.failure_reason == reason
        for call in trace.space_time_planning_calls
    )


def classify_robust_failure(
    scenario: Scenario,
    result: EpisodeResult,
    trace: EpisodeTrace,
) -> str:
    """Assign exactly one deterministic primary failure category.

    Priority order (first match wins):

    1. ``multi_human_egress_conflict`` - the episode collided and every
       collision-egress bridge attempt failed (the planner could not escape
       the colliding humans).
    2. ``human_collision`` - any synchronized robot-human collision.
    3. ``static_goal_unreachable`` - planner evidence of an unreachable goal.
    4. ``no_safe_first_action`` - a grid search failed because no first
       action was safe against the predicted pedestrians.
    5. ``spacetime_search_exhausted`` - the space-time search exhausted its
       open set.
    6. ``time_horizon_exhausted`` - the search reached the frozen time
       horizon without a plan.
    7. ``repeated_replan_failure`` - at least two replan attempts and no
       replan success.
    8. ``no_safe_replan_bridge`` - an initial bridge existed but every
       replan bridge attempt failed.
    9. ``no_safe_initial_bridge`` - no bridge was ever produced.
    10. ``multi_pedestrian_corridor_blockage`` - at least two terminal
        pedestrians jointly occupy a cluster on/near the planned route and
        the robot ended near them.
    11. ``pedestrian_terminal_blockage`` - a terminal pedestrian sits on or
        near the planned route and the robot ended near it.
    12. ``reactive_execution_deadlock`` - exact-zero reactive stalls kept
        the robot deadlocked despite successful replans.
    13. ``progress_stall_timeout`` - progress-stall events with a timeout.
    14. ``other`` - everything else.
    """
    blockage = compute_terminal_blockage_evidence(scenario, trace)
    bridge_successes = trace.continuous_bridge_successes
    replan_successes = trace.robust_replan_successes
    replan_failures = trace.robust_replan_failures
    replan_count = trace.robust_replan_count

    if (
        result.human_collision
        and trace.collision_egress_failures >= 1
        and trace.collision_egress_successes == 0
    ):
        return "multi_human_egress_conflict"
    if result.human_collision:
        return "human_collision"
    if _any_planning_call_failure_reason(trace, "goal_unreachable_static"):
        return "static_goal_unreachable"
    if _any_planning_call_failure_reason(trace, "no_safe_first_action"):
        return "no_safe_first_action"
    if _any_planning_call_failure_reason(trace, "search_exhausted"):
        return "spacetime_search_exhausted"
    if _any_planning_call_failure_reason(trace, "time_horizon_exhausted"):
        return "time_horizon_exhausted"
    if replan_count >= 2 and replan_successes == 0:
        return "repeated_replan_failure"
    if (
        bridge_successes >= 1
        and replan_count >= 1
        and replan_successes == 0
    ):
        return "no_safe_replan_bridge"
    if bridge_successes == 0:
        return "no_safe_initial_bridge"
    if blockage.joint_corridor_blockage and blockage.robot_near_blocker:
        return "multi_pedestrian_corridor_blockage"
    if (
        blockage.terminal_route_blocker_indices
        and blockage.robot_near_blocker
    ):
        return "pedestrian_terminal_blockage"
    if trace.exact_zero_stall_events > 0 and replan_successes > 0:
        return "reactive_execution_deadlock"
    if trace.progress_stall_events > 0:
        return "progress_stall_timeout"
    return "other"


def classify_planning_vs_execution(
    result: EpisodeResult,
    trace: EpisodeTrace,
) -> str:
    """Assign exactly one planning-vs-execution class for a failed episode.

    A. ``planner_no_valid_plan`` - no continuous bridge ever succeeded.
    B. ``planner_ok_execution_deadlock`` - a plan existed, the episode timed
       out with exact-zero reactive stalls, and no collision occurred.
    C. ``execution_progressed_then_collided`` - a plan existed, execution
       progressed, and a later collision occurred (no egress attempt was
       involved, so the collision arose during ordinary execution).
    D. ``repeated_replan_failure_unsafe_state`` - at least two replan
       attempts, all failed, from locally unsafe states.
    E. ``apparently_infeasible_horizon`` - the episode failed within the
       frozen horizon without matching A-D.
    """
    bridge_successes = trace.continuous_bridge_successes
    if bridge_successes == 0:
        return "planner_no_valid_plan"
    if (
        result.human_collision
        and trace.collision_egress_attempts == 0
    ):
        return "execution_progressed_then_collided"
    if (
        trace.robust_replan_failures >= 2
        and trace.robust_replan_successes == 0
        and not result.human_collision
    ):
        return "repeated_replan_failure_unsafe_state"
    if (
        trace.timed_out
        and trace.exact_zero_stall_events > 0
        and not result.human_collision
    ):
        return "planner_ok_execution_deadlock"
    return "apparently_infeasible_horizon"
# ---------------------------------------------------------------------------
# Final-state multi-human geometry
# ---------------------------------------------------------------------------

def compute_final_state_geometry(
    scenario: Scenario,
    result: EpisodeResult,
    trace: EpisodeTrace,
) -> dict[str, object]:
    """Deterministic final-state geometry for one episode.

    Includes per-pedestrian rows, distance-bucket counts around the robot,
    the closest pedestrian, and blocking indices.
    """
    robot_final = trace.final_robot_position
    final_positions = trace.final_pedestrian_positions
    targets = scenario.pedestrians

    distances = tuple(
        hypot(
            robot_final[0] - position[0],
            robot_final[1] - position[1],
        )
        for position in final_positions
    )
    closest_index = (
        min(range(len(distances)), key=lambda index: distances[index])
        if distances
        else None
    )
    per_pedestrian = []
    for index, spec in enumerate(targets):
        position = final_positions[index]
        velocity = compute_pedestrian_velocity(
            position,
            spec.target,
            spec.speed,
        )
        per_pedestrian.append(
            {
                "index": index,
                "final_position": list(position),
                "target": list(spec.target),
                "velocity": list(velocity),
                "speed": spec.speed,
                "minimum_robot_distance": (
                    trace.per_human_minimum_distances[index]
                ),
                "final_robot_distance": distances[index],
                "caused_collision": (
                    index in trace.collision_human_indices
                ),
                "inside_stop_distance_near_failure": (
                    index in trace.blocking_human_indices
                ),
                "terminal_near_failure": pedestrian_at_target(
                    position,
                    spec.target,
                ),
            }
        )

    return {
        "pedestrian_count": trace.pedestrian_count,
        "robot_final_position": list(robot_final),
        "goal_position": [
            scenario.goal[0] * scenario.grid_scale,
            scenario.goal[1] * scenario.grid_scale,
        ],
        "distance_to_goal": hypot(
            robot_final[0] - scenario.goal[0] * scenario.grid_scale,
            robot_final[1] - scenario.goal[1] * scenario.grid_scale,
        ),
        "within_collision_distance": sum(
            distance <= HUMAN_COLLISION_DISTANCE for distance in distances
        ),
        "within_stop_distance": sum(
            distance <= STOP_DISTANCE for distance in distances
        ),
        "within_slow_distance": sum(
            distance <= SLOW_DISTANCE for distance in distances
        ),
        "within_social_distance": sum(
            distance <= SOCIAL_DISTANCE for distance in distances
        ),
        "closest_pedestrian_index": closest_index,
        "blocking_pedestrian_indices": list(trace.blocking_human_indices),
        "collision_human_indices": list(trace.collision_human_indices),
        "per_pedestrian": per_pedestrian,
    }


# ---------------------------------------------------------------------------
# Local action feasibility at the final state
# ---------------------------------------------------------------------------

def _motion_separations_from_pose(
    start_position: Position,
    end_position: Position,
    duration: float,
    pedestrians: tuple[tuple[Position, Position, Position], ...],
    *,
    start_time: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    """Per-human start/mid/end predicted separations for one motion."""
    per_human = []
    for position, velocity, target in pedestrians:
        samples = []
        for fraction in (0.0, 0.5, 1.0):
            robot_at = (
                start_position[0]
                + (end_position[0] - start_position[0]) * fraction,
                start_position[1]
                + (end_position[1] - start_position[1]) * fraction,
            )
            human_at = predict_pedestrian_position_at_time(
                position,
                velocity,
                start_time + duration * fraction,
                target=target,
            )
            samples.append(
                hypot(
                    robot_at[0] - human_at[0],
                    robot_at[1] - human_at[1],
                )
            )
        per_human.append((samples[0], samples[1], samples[2]))
    return tuple(per_human)


def _rejecting_human_indices(
    per_human_separations: tuple[tuple[float, float, float], ...],
    collision_distance: float,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, separations in enumerate(per_human_separations)
        if min(separations) <= collision_distance
    )


def analyze_local_feasibility(
    scenario: Scenario,
    trace: EpisodeTrace,
) -> dict[str, object]:
    """Evaluate UP/RIGHT/DOWN/LEFT/WAIT and robust bridge candidates from
    the robot's final pose against the pedestrians' final states.

    This is pure analysis over the planner's own public primitives; it never
    changes the episode.
    """
    grid_map = build_scenario_grid(scenario)
    robot_position = trace.final_robot_position
    mapped_cell = world_to_nearest_free_cell(
        grid_map,
        robot_position,
        scenario.grid_scale,
    )
    pedestrians = tuple(
        (position, velocity, spec.target)
        for position, velocity, spec in zip(
            trace.final_pedestrian_positions,
            (
                compute_pedestrian_velocity(
                    trace.final_pedestrian_positions[index],
                    spec.target,
                    spec.speed,
                )
                for index, spec in enumerate(scenario.pedestrians)
            ),
            scenario.pedestrians,
        )
    )
    distances = tuple(
        hypot(
            robot_position[0] - position[0],
            robot_position[1] - position[1],
        )
        for position, _, _ in pedestrians
    )
    critical_human_indices = tuple(
        index
        for index, distance in enumerate(distances)
        if distance <= STOP_DISTANCE
    )
    move_duration = scenario.grid_scale / ROBOT_SPEED
    collision_distance = HUMAN_COLLISION_DISTANCE

    action_rows = []
    for action in ACTIONS:
        delta_x, delta_y = _ACTION_OFFSETS[action]
        if action == "WAIT":
            destination_cell = mapped_cell
            end_position = robot_position
        else:
            destination_cell = (
                mapped_cell[0] + delta_x,
                mapped_cell[1] + delta_y,
            )
            end_position = grid_to_world(
                destination_cell,
                scenario.grid_scale,
            )
        inside_map = grid_map.is_inside(destination_cell)
        static_valid = inside_map and grid_map.is_free(destination_cell)
        per_human_separations = _motion_separations_from_pose(
            robot_position,
            end_position,
            move_duration,
            pedestrians,
        )
        overall_minimum = min(
            min(separations) for separations in per_human_separations
        )
        rejecting = _rejecting_human_indices(
            per_human_separations,
            collision_distance,
        )
        humans_for_egress = [
            (position, separations)
            for (position, _, _), separations in zip(
                pedestrians,
                per_human_separations,
            )
        ]
        egress_possible = is_multi_collision_egress_motion_safe(
            robot_position,
            end_position,
            humans_for_egress,
            collision_distance,
        )
        away_from_all_critical = all(
            is_separation_increasing(
                robot_position,
                pedestrians[index][0],
                (
                    end_position[0] - robot_position[0],
                    end_position[1] - robot_position[1],
                ),
                tolerance=DIRECTION_DOT_TOLERANCE,
            )
            for index in critical_human_indices
        )
        action_rows.append(
            {
                "action": action,
                "destination_cell": list(destination_cell),
                "static_valid": static_valid,
                "overall_minimum_separation": overall_minimum,
                "per_human_minimum_separations": [
                    min(separations) for separations in per_human_separations
                ],
                "rejecting_human_indices": list(rejecting),
                "collision_egress_possible": egress_possible,
                "moves_away_from_all_critical_humans": (
                    away_from_all_critical
                ),
                "safe": not rejecting and static_valid,
            }
        )

    closest = None
    if pedestrians:
        closest = min(
            range(len(pedestrians)),
            key=lambda index: distances[index],
        )
    evaluations = build_continuous_start_transitions(
        grid_map,
        robot_position,
        mapped_cell,
        pedestrian_position=(
            pedestrians[closest][0] if closest is not None else None
        ),
        pedestrian_velocity=(
            pedestrians[closest][1] if closest is not None else (0.0, 0.0)
        ),
        pedestrian_target=(
            pedestrians[closest][2] if closest is not None else None
        ),
        grid_scale=scenario.grid_scale,
        robot_speed=ROBOT_SPEED,
        collision_distance=collision_distance,
        additional_pedestrians=tuple(
            state for index, state in enumerate(pedestrians)
            if index != closest
        ),
    )
    bridge_rows = []
    for evaluation in evaluations:
        duration = evaluation.duration
        per_human_separations = _motion_separations_from_pose(
            robot_position,
            evaluation.target_position,
            duration,
            pedestrians,
        )
        rejecting = _rejecting_human_indices(
            per_human_separations,
            collision_distance,
        )
        bridge_rows.append(
            {
                "target_cell": list(evaluation.target_cell),
                "target_position": list(evaluation.target_position),
                "static_valid": evaluation.static_valid,
                "safe": evaluation.safe,
                "rejection_reason": evaluation.rejection_reason,
                "overall_minimum_separation": (
                    evaluation.minimum_predicted_separation
                ),
                "per_human_minimum_separations": [
                    min(separations) for separations in per_human_separations
                ],
                "rejecting_human_indices": list(rejecting),
                "collision_egress_possible": (
                    is_multi_collision_egress_motion_safe(
                        robot_position,
                        evaluation.target_position,
                        [
                            (position, separations)
                            for (position, _, _), separations in zip(
                                pedestrians,
                                per_human_separations,
                            )
                        ],
                        collision_distance,
                    )
                ),
            }
        )

    any_safe_bridge = any(
        row["safe"] is True for row in bridge_rows
    )
    any_safe_action = any(
        row["safe"] is True for row in action_rows
    )
    return {
        "mapped_cell": list(mapped_cell),
        "critical_human_indices": list(critical_human_indices),
        "actions": action_rows,
        "bridge_candidates": bridge_rows,
        "locally_trapped": not any_safe_bridge and not any_safe_action,
        "no_safe_bridge": not any_safe_bridge,
        "no_safe_action": not any_safe_action,
        "multi_human_nearby": sum(
            distance <= SLOW_DISTANCE for distance in distances
        )
        >= 2,
    }
# ---------------------------------------------------------------------------
# Collision attribution (uses probe-recorded trajectories and phases)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeEvidence:
    """Recorded trajectories and per-step phases from the evidence probe."""

    robot_trajectory: tuple[Position, ...]
    human_trajectories: tuple[tuple[Position, ...], ...]
    phases: tuple[str, ...]


def analyze_first_collision(
    scenario: Scenario,
    result: EpisodeResult,
    trace: EpisodeTrace,
    evidence: EpisodeEvidence,
    *,
    collision_distance: float = HUMAN_COLLISION_DISTANCE,
    dt: float = SIMULATION_STEP,
) -> dict[str, object] | None:
    """Attribute the first collision of an episode using recorded evidence.

    Returns ``None`` when the episode has no collision.  The first collision
    timestep, colliding pedestrian indices, execution phase, robot speed
    scale, relative motion directions, and whether a pedestrian walked into
    a stopped robot are all derived deterministically from the recorded
    trajectories.
    """
    if not result.human_collision:
        return None

    robot = evidence.robot_trajectory
    humans = evidence.human_trajectories
    first_step = None
    first_indices: tuple[int, ...] = ()
    for step in range(1, len(robot)):
        indices = tuple(
            index
            for index, human_trajectory in enumerate(humans)
            if hypot(
                robot[step][0] - human_trajectory[step][0],
                robot[step][1] - human_trajectory[step][1],
            )
            <= collision_distance
        )
        if indices:
            first_step = step
            first_indices = indices
            break
    if first_step is None:
        return None

    phase = (
        evidence.phases[first_step]
        if first_step < len(evidence.phases)
        else "unknown"
    )
    speed_scale = (
        trace.speed_scales[first_step - 1]
        if first_step - 1 < len(trace.speed_scales)
        else None
    )
    per_human = []
    for index in first_indices:
        robot_motion = (
            robot[first_step][0] - robot[first_step - 1][0],
            robot[first_step][1] - robot[first_step - 1][1],
        )
        human_motion = (
            humans[index][first_step][0]
            - humans[index][first_step - 1][0],
            humans[index][first_step][1]
            - humans[index][first_step - 1][1],
        )
        to_human_before = (
            humans[index][first_step - 1][0] - robot[first_step - 1][0],
            humans[index][first_step - 1][1] - robot[first_step - 1][1],
        )
        robot_toward_human = (
            hypot(*robot_motion) > 0.0
            and hypot(*to_human_before) > 0.0
            and (
                robot_motion[0] * to_human_before[0]
                + robot_motion[1] * to_human_before[1]
            )
            > DIRECTION_DOT_TOLERANCE
        )
        to_robot_before = (
            -to_human_before[0],
            -to_human_before[1],
        )
        human_toward_robot = (
            hypot(*human_motion) > 0.0
            and hypot(*to_robot_before) > 0.0
            and (
                human_motion[0] * to_robot_before[0]
                + human_motion[1] * to_robot_before[1]
            )
            > DIRECTION_DOT_TOLERANCE
        )
        separation_before = hypot(
            robot[first_step - 1][0] - humans[index][first_step - 1][0],
            robot[first_step - 1][1] - humans[index][first_step - 1][1],
        )
        # Measurable "unavoidable" evidence: using the pre-collision state
        # and the pedestrian's own deterministic motion model, check whether
        # the pedestrian's predicted trajectory alone reaches within the
        # collision distance of the robot's next actual position.
        human_on_collision_course = any(
            hypot(
                robot[first_step][0] - predicted_position[0],
                robot[first_step][1] - predicted_position[1],
            )
            <= collision_distance
            for predicted_position in (
                predict_pedestrian_position_at_time(
                    humans[index][first_step - 1],
                    compute_pedestrian_velocity(
                        humans[index][first_step - 1],
                        scenario.pedestrians[index].target,
                        scenario.pedestrians[index].speed,
                    ),
                    dt * fraction,
                    target=scenario.pedestrians[index].target,
                )
                for fraction in (0.5, 1.0)
            )
        )
        per_human.append(
            {
                "pedestrian_index": index,
                "separation_before_step": separation_before,
                "robot_moved_toward_human": robot_toward_human,
                "human_moved_toward_robot": human_toward_robot,
                "human_on_collision_course": human_on_collision_course,
            }
        )

    return {
        "first_collision_step": first_step,
        "first_collision_seconds": first_step * dt,
        "colliding_pedestrian_indices": list(first_indices),
        "phase": phase,
        "robot_speed_scale": speed_scale,
        "robot_stopped": speed_scale is not None and speed_scale == 0.0,
        "pedestrian_moved_into_robot": (
            speed_scale is not None
            and speed_scale == 0.0
            and any(
                row["human_moved_toward_robot"] for row in per_human
            )
        ),
        "per_human": per_human,
    }


# ---------------------------------------------------------------------------
# Scenario difficulty features
# ---------------------------------------------------------------------------

def scenario_difficulty_features(
    scenario: Scenario,
    *,
    social_distance: float = SOCIAL_DISTANCE,
    horizon_seconds: float = 20.0,
    route_samples: int = 100,
    time_samples: int = 200,
) -> dict[str, object]:
    """Deterministic geometric difficulty features for one scenario.

    No learned model is involved: the robot A* route and the pedestrians'
    straight-line motion models are combined with simple geometry.
    """
    grid_map = build_scenario_grid(scenario)
    path = astar(grid_map, scenario.start, scenario.goal)
    if path is None:
        raise ValueError("scenario must have a reachable A* route")
    route = tuple(
        grid_to_world(coordinate, scenario.grid_scale)
        for coordinate in path
    )

    specs = scenario.pedestrians
    start_route_distances = tuple(
        point_to_route_distance(spec.start, route) for spec in specs
    )
    target_route_distances = tuple(
        point_to_route_distance(spec.target, route) for spec in specs
    )
    crossing = tuple(
        segment_intersects_route(spec.start, spec.target, route)
        for spec in specs
    )
    approaching = tuple(
        min(
            point_to_route_distance(spec.start, route),
            point_to_route_distance(spec.target, route),
        )
        <= social_distance
        for spec in specs
    )

    # Approximate maximum simultaneous humans near the route: sample each
    # pedestrian's clamped straight-line trajectory in time and sweep.
    interval_by_pedestrian = []
    for spec in specs:
        near_samples = []
        for sample_index in range(time_samples + 1):
            time = horizon_seconds * sample_index / time_samples
            position = predict_pedestrian_position_at_time(
                spec.start,
                compute_pedestrian_velocity(
                    spec.start,
                    spec.target,
                    spec.speed,
                ),
                time,
                target=spec.target,
            )
            if point_to_route_distance(position, route) <= social_distance:
                near_samples.append(time)
        if near_samples:
            interval_by_pedestrian.append(
                (near_samples[0], near_samples[-1])
            )
    events = []
    for start_time, end_time in interval_by_pedestrian:
        events.append((start_time, 1))
        events.append((end_time, -1))
    events.sort(key=lambda event: (event[0], -event[1]))
    maximum_simultaneous = 0
    current = 0
    for _, delta in events:
        current += delta
        maximum_simultaneous = max(maximum_simultaneous, current)

    return {
        "pedestrian_count": len(specs),
        "astar_path_length_moves": len(path) - 1,
        "astar_path_length_world": (
            (len(path) - 1) * scenario.grid_scale
        ),
        "trajectories_intersecting_or_approaching_route": sum(
            crossing[index] or approaching[index]
            for index in range(len(specs))
        ),
        "minimum_initial_human_route_distance": (
            min(start_route_distances) if start_route_distances else None
        ),
        "route_crossing_pedestrians": sum(crossing),
        "pedestrian_targets_near_route": sum(
            distance <= social_distance
            for distance in target_route_distances
        ),
        "approximate_max_simultaneous_nearby_humans": maximum_simultaneous,
    }


def aggregate_features(
    rows: list[dict[str, object]],
    numeric_keys: tuple[str, ...],
) -> dict[str, float]:
    """Mean/min/max of numeric features over a group of episode rows."""
    aggregates = {}
    for key in numeric_keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue
        aggregates[f"mean_{key}"] = mean(values)
        aggregates[f"min_{key}"] = min(values)
        aggregates[f"max_{key}"] = max(values)
    return aggregates


# ---------------------------------------------------------------------------
# Trace counter and search-statistics aggregation
# ---------------------------------------------------------------------------

def summarize_trace_counters(
    results: list[EpisodeResult],
    traces: list[EpisodeTrace],
) -> dict[str, object]:
    """Mean/max robust counters split into failing and succeeding episodes."""
    counter_keys = (
        "continuous_bridge_attempts",
        "continuous_bridge_failures",
        "robust_replan_count",
        "robust_replan_successes",
        "robust_replan_failures",
        "exact_zero_stall_events",
        "progress_stall_events",
        "suppressed_duplicate_replans",
        "collision_egress_attempts",
        "collision_egress_failures",
    )
    groups = {"failures": [], "successes": []}
    for result, trace in zip(results, traces):
        group = "successes" if result.success else "failures"
        groups[group].append(trace)

    summary: dict[str, object] = {}
    for group_name, group_traces in groups.items():
        group_summary = {}
        for key in counter_keys:
            values = [getattr(trace, key) for trace in group_traces]
            group_summary[key] = {
                "total": sum(values),
                "mean": mean(values) if values else None,
                "max": max(values) if values else None,
            }
        group_summary["episodes"] = len(group_traces)
        summary[group_name] = group_summary
    return summary


def summarize_search_statistics(
    results: list[EpisodeResult],
    traces: list[EpisodeTrace],
) -> dict[str, object]:
    """Mean/median/p90/max expanded and generated states per planning call,
    planning-call counts, and planning failures, split by success."""
    def quantiles(values: list[int]) -> dict[str, float | None]:
        if not values:
            return {
                "mean": None,
                "median": None,
                "p90": None,
                "max": None,
            }
        ordered = sorted(values)
        p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
        return {
            "mean": mean(ordered),
            "median": median(ordered),
            "p90": ordered[p90_index],
            "max": ordered[-1],
        }

    groups = {"failures": [], "successes": [], "all": []}
    for result, trace in zip(results, traces):
        expanded = [
            call.expanded_states for call in trace.space_time_planning_calls
        ]
        generated = [
            call.generated_states for call in trace.space_time_planning_calls
        ]
        record = {
            "expanded": expanded,
            "generated": generated,
            "planning_calls": len(expanded),
            "planning_failures": trace.spacetime_planning_failures,
        }
        groups["all"].append(record)
        groups[
            "successes" if result.success else "failures"
        ].append(record)

    summary: dict[str, object] = {}
    for group_name, records in groups.items():
        expanded = [
            value for record in records for value in record["expanded"]
        ]
        generated = [
            value for record in records for value in record["generated"]
        ]
        call_counts = [record["planning_calls"] for record in records]
        failures = [record["planning_failures"] for record in records]
        summary[group_name] = {
            "episodes": len(records),
            "expanded_states": quantiles(expanded),
            "generated_states": quantiles(generated),
            "planning_calls": {
                "total": sum(call_counts),
                "mean": mean(call_counts) if call_counts else None,
                "max": max(call_counts) if call_counts else None,
            },
            "planning_failures": {
                "total": sum(failures),
                "mean": mean(failures) if failures else None,
                "max": max(failures) if failures else None,
            },
        }
    return summary


# ---------------------------------------------------------------------------
# Representative-case selection
# ---------------------------------------------------------------------------

_TERMINAL_BLOCKAGE_CATEGORIES = (
    "pedestrian_terminal_blockage",
    "multi_pedestrian_corridor_blockage",
)


def select_representative_cases(
    episode_rows: list[dict[str, object]],
    requirements: dict[int, list[dict[str, str]]],
) -> dict[str, dict[str, object]]:
    """Pick the lowest unused scenario id satisfying each (density, kind) rule.

    ``requirements`` maps a pedestrian count to selectors; each selector
    carries a ``kind`` (e.g. ``collision``, ``timeout``, ``local_trap``,
    ``corridor_blockage``, ``planning_failure``, ``terminal_blockage``,
    ``different_dominant``) and a ``label``.

    For the failure-mode kinds (``corridor_blockage`` and
    ``terminal_blockage``), episodes whose dominant failure category is the
    mode itself are preferred; purely evidence-based matches are used only
    as a fallback when no dominant-category episode exists.  Within one
    density each episode may serve at most one selector, so the
    representative set stays distinct.
    """
    def matches(
        row: dict[str, object],
        kind: str,
        *,
        require_category: bool,
    ) -> bool:
        category = row.get("failure_category")
        timed_out = bool(row.get("timed_out"))
        if kind == "collision":
            return category in (
                "human_collision",
                "multi_human_egress_conflict",
            )
        if kind == "timeout":
            return timed_out
        if kind == "corridor_blockage":
            if require_category:
                return category == "multi_pedestrian_corridor_blockage"
            blockage = row.get("terminal_blockage")
            return bool(
                blockage
                and blockage.get("joint_corridor_blockage") is True
            )
        if kind == "planning_failure":
            return category in (
                "no_safe_first_action",
                "spacetime_search_exhausted",
                "time_horizon_exhausted",
                "repeated_replan_failure",
                "no_safe_replan_bridge",
                "no_safe_initial_bridge",
                "static_goal_unreachable",
            )
        if kind == "local_trap":
            return bool(row.get("locally_trapped"))
        if kind == "terminal_blockage":
            if require_category:
                return category in _TERMINAL_BLOCKAGE_CATEGORIES
            blockage = row.get("terminal_blockage")
            return bool(
                blockage
                and blockage.get("terminal_route_blocker_indices")
            )
        if kind == "different_dominant":
            return (
                category is not None
                and category not in (
                    "human_collision",
                    "multi_human_egress_conflict",
                    *_TERMINAL_BLOCKAGE_CATEGORIES,
                )
                and not bool(row.get("locally_trapped"))
            )
        raise ValueError(f"unknown representative selector {kind!r}")

    selected: dict[str, dict[str, object]] = {}
    for density, selectors in requirements.items():
        used_ids: set[str] = set()
        for selector in selectors:
            kind = selector["kind"]
            label = selector["label"]
            key = f"n{density}_{label}"
            chosen = None
            for require_category in (True, False):
                candidates = [
                    row
                    for row in episode_rows
                    if int(row["density"]) == density
                    and row["scenario_id"] not in used_ids
                    and matches(
                        row,
                        kind,
                        require_category=require_category,
                    )
                ]
                if candidates:
                    chosen = min(
                        candidates,
                        key=lambda row: row["scenario_id"],
                    )
                    break
            if chosen is None:
                selected[key] = {"selected": None}
                continue
            used_ids.add(chosen["scenario_id"])
            selected[key] = {
                "selected": chosen["scenario_id"],
                "kind": kind,
                "density": density,
            }
    return selected
