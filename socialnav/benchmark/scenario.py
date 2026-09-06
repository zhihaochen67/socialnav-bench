"""Reproducible deterministic multi-pedestrian benchmark scenarios."""

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
class PedestrianSpec:
    """Immutable independent straight-line motion for one pedestrian."""

    start: Position
    target: Position
    speed: float


@dataclass(frozen=True)
class Scenario:
    """All static inputs required to reproduce one benchmark episode.

    ``pedestrians`` is the single source of truth for the pedestrian
    population.  For backwards compatibility, single-human scenarios may
    still be constructed with the legacy ``pedestrian_start``,
    ``pedestrian_target``, and ``pedestrian_speed`` keyword arguments;
    they are folded into a one-element ``pedestrians`` tuple.
    """

    scenario_id: str
    grid_width: int
    grid_height: int
    obstacle_cells: tuple[Coordinate, ...]
    start: Coordinate
    goal: Coordinate
    grid_scale: float
    pedestrians: tuple[PedestrianSpec, ...] = ()

    def __init__(
        self,
        scenario_id: str,
        grid_width: int,
        grid_height: int,
        obstacle_cells: tuple[Coordinate, ...],
        start: Coordinate,
        goal: Coordinate,
        grid_scale: float,
        pedestrians: tuple[PedestrianSpec, ...] = (),
        *,
        pedestrian_start: Position | None = None,
        pedestrian_target: Position | None = None,
        pedestrian_speed: float | None = None,
    ) -> None:
        legacy_values = (
            pedestrian_start,
            pedestrian_target,
            pedestrian_speed,
        )
        legacy_supplied = any(value is not None for value in legacy_values)
        if pedestrians and legacy_supplied:
            raise ValueError(
                "pass either pedestrians= or the legacy pedestrian_start/"
                "pedestrian_target/pedestrian_speed keywords, not both"
            )
        if legacy_supplied:
            if not all(value is not None for value in legacy_values):
                raise ValueError(
                    "pedestrian_start, pedestrian_target, and "
                    "pedestrian_speed must be supplied together"
                )
            assert pedestrian_start is not None
            assert pedestrian_target is not None
            assert pedestrian_speed is not None
            pedestrians = (
                PedestrianSpec(
                    start=pedestrian_start,
                    target=pedestrian_target,
                    speed=pedestrian_speed,
                ),
            )
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "grid_width", grid_width)
        object.__setattr__(self, "grid_height", grid_height)
        object.__setattr__(self, "obstacle_cells", tuple(obstacle_cells))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "grid_scale", grid_scale)
        object.__setattr__(self, "pedestrians", tuple(pedestrians))

    @property
    def pedestrian_count(self) -> int:
        """Number of pedestrians in the scenario."""
        return len(self.pedestrians)

    @property
    def pedestrian_start(self) -> Position:
        """Start of the single pedestrian; only valid for N=1 scenarios."""
        return self.pedestrians[0].start

    @property
    def pedestrian_target(self) -> Position:
        """Target of the single pedestrian; only valid for N=1 scenarios."""
        return self.pedestrians[0].target

    @property
    def pedestrian_speed(self) -> float:
        """Speed of the single pedestrian; only valid for N=1 scenarios."""
        return self.pedestrians[0].speed


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


def _generate_single_controlled_scenarios(
    count: int,
    seed: int,
) -> list[Scenario]:
    """The original single-human controlled generator, byte-for-byte."""
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


def _generate_single_diverse_scenarios(
    count: int,
    seed: int,
) -> list[Scenario]:
    """The original single-human diverse generator, byte-for-byte."""
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


def _select_pedestrian_specs(
    grid_map: GridMap,
    path: list[Coordinate],
    start: Coordinate,
    goal: Coordinate,
    pedestrian_count: int,
    random: Random,
    *,
    min_speed: float,
    max_speed: float,
) -> tuple[PedestrianSpec, ...]:
    """Deterministically select non-overlapping independent pedestrian specs.

    The candidate pool consists of A*-route cells plus their four-connected
    neighbours, excluding the robot start and goal cells.  When the pool is
    too small to hold ``2 * pedestrian_count`` distinct cells the pool falls
    back to every free cell (still excluding the robot endpoints).  Selection
    uses only the seeded local RNG and ordinary A* - no social, predictive,
    space-time, or robust planner participates in acceptance.
    """
    if pedestrian_count == 0:
        return ()
    if pedestrian_count < 0:
        raise ValueError("pedestrian_count must be non-negative")

    excluded = {start, goal}
    path_cells = set(path)
    near_path_cells = {
        (cell[0] + delta_x, cell[1] + delta_y)
        for cell in path_cells
        for delta_x, delta_y in _NEIGHBOR_OFFSETS
    }
    pool = sorted(
        cell
        for cell in (path_cells | near_path_cells)
        if cell not in excluded and grid_map.is_free(cell)
    )
    if len(pool) < 2 * pedestrian_count:
        pool = sorted(
            cell for cell in _free_cells(grid_map) if cell not in excluded
        )
    if len(pool) < 2 * pedestrian_count:
        raise RuntimeError(
            f"scenario map has too few free cells to place "
            f"{pedestrian_count} pedestrians"
        )

    shuffled = pool[:]
    random.shuffle(shuffled)
    start_cells = shuffled[:pedestrian_count]
    target_cells = shuffled[pedestrian_count : 2 * pedestrian_count]
    specs = []
    for pedestrian_start_cell, pedestrian_target_cell in zip(
        start_cells,
        target_cells,
    ):
        speed = round(random.uniform(min_speed, max_speed), 6)
        specs.append(
            PedestrianSpec(
                start=grid_to_world(pedestrian_start_cell),
                target=grid_to_world(pedestrian_target_cell),
                speed=speed,
            )
        )
    return tuple(specs)


def _generate_multi_human_scenarios(
    count: int,
    seed: int,
    pedestrian_count: int,
    *,
    controlled: bool,
) -> list[Scenario]:
    """Deterministic multi-human generation shared by both scenario modes.

    Robot endpoints come from the fixed controlled map or from the
    planner-neutral diverse templates.  Pedestrian selection is
    planner-neutral: only ordinary A* is consulted, and only to describe
    which cells are near the robot's route.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if pedestrian_count < 0:
        raise ValueError("pedestrian_count must be non-negative")

    random = Random(seed)
    if controlled:
        min_speed = _MIN_PEDESTRIAN_SPEED
        max_speed = _MAX_PEDESTRIAN_SPEED
        scenario_id_prefix = "seed"
    else:
        min_speed = _DIVERSE_MIN_PEDESTRIAN_SPEED
        max_speed = _DIVERSE_MAX_PEDESTRIAN_SPEED
        scenario_id_prefix = "diverse-seed"

    templates: list[
        tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    ] | None = None
    remaining_templates: list[
        tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    ] = []
    scenarios = []

    for episode_index in range(count):
        if controlled:
            start, goal = START, GOAL
        else:
            if templates is None:
                templates = _diverse_options()
            if not remaining_templates:
                remaining_templates = templates.copy()
                random.shuffle(remaining_templates)
            start, goal, _, _ = remaining_templates.pop()

        grid_map = build_demo_grid()
        path = astar(grid_map, start, goal)
        if path is None:
            raise RuntimeError("scenario endpoints must have an A* path")
        specs = _select_pedestrian_specs(
            grid_map,
            path,
            start,
            goal,
            pedestrian_count,
            random,
            min_speed=min_speed,
            max_speed=max_speed,
        )
        scenarios.append(
            Scenario(
                scenario_id=(
                    f"{scenario_id_prefix}-{seed}-episode-"
                    f"{episode_index:04d}-pedestrians-{pedestrian_count}"
                ),
                grid_width=GRID_WIDTH,
                grid_height=GRID_HEIGHT,
                obstacle_cells=tuple(sorted(OBSTACLES)),
                start=start,
                goal=goal,
                grid_scale=CELL_SIZE,
                pedestrians=specs,
            )
        )

    return scenarios


def generate_scenarios(
    count: int,
    seed: int,
    pedestrian_count: int = 1,
) -> list[Scenario]:
    """Generate seeded scenarios with the requested deterministic pedestrian count.

    ``pedestrian_count=1`` uses the original single-human generator unchanged.
    """
    if pedestrian_count == 1:
        return _generate_single_controlled_scenarios(count, seed)
    return _generate_multi_human_scenarios(
        count,
        seed,
        pedestrian_count,
        controlled=True,
    )


def generate_diverse_scenarios(
    count: int,
    seed: int,
    pedestrian_count: int = 1,
) -> list[Scenario]:
    """Generate seeded diverse scenarios with the requested pedestrian count.

    ``pedestrian_count=1`` uses the original single-human diverse generator
    unchanged.
    """
    if pedestrian_count == 1:
        return _generate_single_diverse_scenarios(count, seed)
    return _generate_multi_human_scenarios(
        count,
        seed,
        pedestrian_count,
        controlled=False,
    )


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