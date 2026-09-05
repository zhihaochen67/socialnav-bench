"""Deterministic helpers for event-triggered online social replanning."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot

from socialnav.env.demo_map import grid_to_world
from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.env.world import SIMULATION_STEP, STEPS_PER_CELL
from socialnav.metrics import Position

REPLAN_STOP_SECONDS = 0.50
REPLAN_STOP_STEPS = ceil(REPLAN_STOP_SECONDS / SIMULATION_STEP)


@dataclass
class SustainedStopReplanPolicy:
    """Trigger after each fresh run of consecutive zero-speed steps.

    Resetting the counter on a trigger is the anti-thrashing policy: another
    trigger requires a complete new sustained-stop interval.
    """

    threshold_steps: int = REPLAN_STOP_STEPS
    consecutive_stopped_steps: int = 0

    def __post_init__(self) -> None:
        if self.threshold_steps <= 0:
            raise ValueError("threshold_steps must be positive")

    def observe(self, speed_scale: float) -> bool:
        """Observe one control output and report whether to replan now."""
        if speed_scale != 0.0:
            self.consecutive_stopped_steps = 0
            return False

        self.consecutive_stopped_steps += 1
        if self.consecutive_stopped_steps < self.threshold_steps:
            return False

        self.consecutive_stopped_steps = 0
        return True


def world_to_nearest_free_cell(
    grid_map: GridMap,
    position: Position,
    grid_scale: float,
) -> Coordinate:
    """Map a world position to its nearest free grid-cell centre.

    Distance is measured in world space. Exact ties choose the lower grid x,
    then the lower grid y. Searching only free cells also clamps positions
    outside the map to a valid free boundary cell without changing the map.
    """
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")

    free_cells = (
        (x, y)
        for x in range(grid_map.width)
        for y in range(grid_map.height)
        if grid_map.is_free((x, y))
    )
    try:
        return min(
            free_cells,
            key=lambda coordinate: (
                (
                    grid_to_world(coordinate, grid_scale)[0] - position[0]
                )
                ** 2
                + (
                    grid_to_world(coordinate, grid_scale)[1] - position[1]
                )
                ** 2,
                coordinate[0],
                coordinate[1],
            ),
        )
    except ValueError as error:
        raise ValueError("grid map must contain a free cell") from error


def interpolate_replanned_route(
    current_position: Position,
    grid_path: list[Coordinate],
    grid_scale: float,
    *,
    steps_per_cell: int = STEPS_PER_CELL,
) -> list[Position]:
    """Join the physical robot position to a newly planned grid route.

    The first returned point is always the robot's unchanged current world
    position. Subsequent points approach the mapped start-cell centre and then
    follow the grid route, with spacing no larger than the normal benchmark
    path spacing.
    """
    if not grid_path:
        raise ValueError("grid_path must not be empty")
    if grid_scale <= 0.0:
        raise ValueError("grid_scale must be positive")
    if steps_per_cell <= 0:
        raise ValueError("steps_per_cell must be positive")

    maximum_step_distance = grid_scale / steps_per_cell
    waypoints = [
        current_position,
        *(
            grid_to_world(coordinate, grid_scale)
            for coordinate in grid_path
        ),
    ]
    positions = [current_position]

    for start, end in zip(waypoints, waypoints[1:]):
        distance = hypot(end[0] - start[0], end[1] - start[1])
        if distance == 0.0:
            continue
        segment_steps = max(1, ceil(distance / maximum_step_distance))
        for step in range(1, segment_steps + 1):
            fraction = step / segment_steps
            positions.append(
                (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            )

    return positions
