"""Reproducible single-pedestrian benchmark scenarios."""

from dataclasses import dataclass
from random import Random

from socialnav.env.demo_map import (
    CELL_SIZE,
    GOAL,
    GRID_HEIGHT,
    GRID_WIDTH,
    OBSTACLES,
    SOCIAL_DISTANCE,
    SOCIAL_WEIGHT,
    START,
    build_demo_grid,
    grid_to_world,
)
from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.metrics import Position
from socialnav.planners.astar import astar
from socialnav.planners.social_planner import social_astar

_MIN_PEDESTRIAN_SPEED = 0.10
_MAX_PEDESTRIAN_SPEED = 0.16
_TARGET_OFFSET_CELLS = 2
DIVERSE_MIN_PATH_MOVES = 6
_DIVERSE_MIN_PEDESTRIAN_SPEED = 0.20
_DIVERSE_MAX_PEDESTRIAN_SPEED = 0.60
_NEIGHBOR_OFFSETS: tuple[Coordinate, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


@dataclass(frozen=True)
class Scenario:
    """All static inputs required to reproduce one benchmark episode."""

    scenario_id: str
    grid_width: int
    grid_height: int
    obstacle_cells: tuple[Coordinate, ...]
    start: Coordinate
    goal: Coordinate
    grid_scale: float
    pedestrian_start: Position
    pedestrian_target: Position
    pedestrian_speed: float


def build_scenario_grid(scenario: Scenario) -> GridMap:
    """Build a fresh occupancy grid from a scenario definition."""
    grid_map = GridMap(scenario.grid_width, scenario.grid_height)
    for obstacle in scenario.obstacle_cells:
        grid_map.add_obstacle(obstacle)
    return grid_map


def _crossing_options() -> list[tuple[Coordinate, Coordinate]]:
    grid_map = build_demo_grid()
    path = astar(grid_map, START, GOAL)
    if path is None:
        raise RuntimeError("benchmark base map must have an A* path")

    path_cells = set(path)
    options: list[tuple[Coordinate, Coordinate]] = []
    for path_index, pedestrian_cell in enumerate(path):
        if path_index < 4 or path_index > len(path) // 2:
            continue

        social_path = social_astar(
            grid_map,
            START,
            GOAL,
            pedestrian_positions=[grid_to_world(pedestrian_cell)],
            social_distance=SOCIAL_DISTANCE,
            social_weight=SOCIAL_WEIGHT,
            grid_scale=CELL_SIZE,
        )
        if social_path is None or len(social_path) <= len(path):
            continue

        for dx, dy in _NEIGHBOR_OFFSETS:
            target_cell = (
                pedestrian_cell[0] + dx,
                pedestrian_cell[1] + dy,
            )
            if (
                grid_map.is_free(target_cell)
                and target_cell not in path_cells
                and target_cell not in social_path
            ):
                options.append((pedestrian_cell, target_cell))

    if not options:
        raise RuntimeError("benchmark base map has no relevant crossing options")
    return options


def generate_scenarios(count: int, seed: int) -> list[Scenario]:
    """Generate seeded scenarios whose pedestrian starts on the A* route."""
    if count < 0:
        raise ValueError("count must be non-negative")

    random = Random(seed)
    crossing_options = _crossing_options()
    scenarios = []

    for episode_index in range(count):
        pedestrian_cell, target_cell = random.choice(crossing_options)
        pedestrian_speed = round(
            random.uniform(_MIN_PEDESTRIAN_SPEED, _MAX_PEDESTRIAN_SPEED),
            6,
        )
        pedestrian_start = grid_to_world(pedestrian_cell)
        target_offset = (
            target_cell[0] - pedestrian_cell[0],
            target_cell[1] - pedestrian_cell[1],
        )
        scenarios.append(
            Scenario(
                scenario_id=f"seed-{seed}-episode-{episode_index:04d}",
                grid_width=GRID_WIDTH,
                grid_height=GRID_HEIGHT,
                obstacle_cells=tuple(sorted(OBSTACLES)),
                start=START,
                goal=GOAL,
                grid_scale=CELL_SIZE,
                pedestrian_start=pedestrian_start,
                pedestrian_target=(
                    pedestrian_start[0]
                    + target_offset[0] * CELL_SIZE * _TARGET_OFFSET_CELLS,
                    pedestrian_start[1]
                    + target_offset[1] * CELL_SIZE * _TARGET_OFFSET_CELLS,
                ),
                pedestrian_speed=pedestrian_speed,
            )
        )

    return scenarios


def _free_cells(grid_map: GridMap) -> list[Coordinate]:
    return [
        (x, y)
        for x in range(grid_map.width)
        for y in range(grid_map.height)
        if grid_map.is_free((x, y))
    ]


def _off_route_neighbors(
    grid_map: GridMap,
    waypoint: Coordinate,
    path_cells: set[Coordinate],
) -> list[Coordinate]:
    neighbors = [
        (waypoint[0] + dx, waypoint[1] + dy)
        for dx, dy in _NEIGHBOR_OFFSETS
    ]
    return [
        neighbor
        for neighbor in neighbors
        if grid_map.is_free(neighbor) and neighbor not in path_cells
    ]


def _diverse_options() -> list[
    tuple[Coordinate, Coordinate, Coordinate, Coordinate]
]:
    """Return planner-neutral start, goal, and pedestrian templates."""
    grid_map = build_demo_grid()
    free_cells = _free_cells(grid_map)
    options: list[tuple[Coordinate, Coordinate, Coordinate, Coordinate]] = []

    for start in free_cells:
        for goal in free_cells:
            if start == goal:
                continue

            path = astar(grid_map, start, goal)
            if path is None or len(path) - 1 < DIVERSE_MIN_PATH_MOVES:
                continue

            path_cells = set(path)
            for waypoint in path[1:-1]:
                off_route = _off_route_neighbors(
                    grid_map,
                    waypoint,
                    path_cells,
                )

                for pedestrian_target in off_route:
                    options.append(
                        (start, goal, waypoint, pedestrian_target)
                    )

                for pedestrian_start in off_route:
                    for pedestrian_target in off_route:
                        if pedestrian_start != pedestrian_target:
                            options.append(
                                (
                                    start,
                                    goal,
                                    pedestrian_start,
                                    pedestrian_target,
                                )
                            )

    if not options:
        raise RuntimeError("benchmark map has no diverse scenario options")
    return options


def generate_diverse_scenarios(count: int, seed: int) -> list[Scenario]:
    """Generate seeded planner-neutral scenarios around ordinary A* paths."""
    if count < 0:
        raise ValueError("count must be non-negative")

    random = Random(seed)
    options = _diverse_options()
    remaining_options: list[
        tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    ] = []
    scenarios = []

    for episode_index in range(count):
        if not remaining_options:
            remaining_options = options.copy()
            random.shuffle(remaining_options)

        start, goal, pedestrian_start, pedestrian_target = (
            remaining_options.pop()
        )
        pedestrian_speed = round(
            random.uniform(
                _DIVERSE_MIN_PEDESTRIAN_SPEED,
                _DIVERSE_MAX_PEDESTRIAN_SPEED,
            ),
            6,
        )
        scenarios.append(
            Scenario(
                scenario_id=(
                    f"diverse-seed-{seed}-episode-{episode_index:04d}"
                ),
                grid_width=GRID_WIDTH,
                grid_height=GRID_HEIGHT,
                obstacle_cells=tuple(sorted(OBSTACLES)),
                start=start,
                goal=goal,
                grid_scale=CELL_SIZE,
                pedestrian_start=grid_to_world(pedestrian_start),
                pedestrian_target=grid_to_world(pedestrian_target),
                pedestrian_speed=pedestrian_speed,
            )
        )

    return scenarios


def is_pedestrian_route_relevant(
    scenario: Scenario,
    path: list[Coordinate],
) -> bool:
    """Return whether pedestrian endpoints touch one route neighborhood."""
    pedestrian_start = (
        round(scenario.pedestrian_start[0] / scenario.grid_scale),
        round(scenario.pedestrian_start[1] / scenario.grid_scale),
    )
    pedestrian_target = (
        round(scenario.pedestrian_target[0] / scenario.grid_scale),
        round(scenario.pedestrian_target[1] / scenario.grid_scale),
    )

    for waypoint in path[1:-1]:
        start_distance = abs(pedestrian_start[0] - waypoint[0]) + abs(
            pedestrian_start[1] - waypoint[1]
        )
        target_distance = abs(pedestrian_target[0] - waypoint[0]) + abs(
            pedestrian_target[1] - waypoint[1]
        )
        if start_distance <= 1 and target_distance == 1:
            return True

    return False
