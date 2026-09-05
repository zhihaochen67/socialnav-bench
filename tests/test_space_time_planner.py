from socialnav.env.grid_map import GridMap
from socialnav.planners.space_time_planner import (
    compute_time_aligned_social_cost,
    duration_to_simulation_steps,
    space_time_social_astar,
)


def _plan(
    grid_map: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    pedestrian_position: tuple[float, float] = (10.0, 10.0),
    pedestrian_velocity: tuple[float, float] = (0.0, 0.0),
    pedestrian_target: tuple[float, float] | None = (10.0, 10.0),
    collision_distance: float = 0.34,
    max_time_seconds: float = 10.0,
):
    return space_time_social_astar(
        grid_map,
        start,
        goal,
        pedestrian_position,
        pedestrian_velocity,
        pedestrian_target,
        0.7,
        10.0,
        collision_distance,
        1.0,
        1.0,
        max_time_seconds,
    )


def test_duration_rounding_is_nearest_with_ties_up() -> None:
    assert duration_to_simulation_steps(0.375, 1.0 / 240.0) == 90
    assert duration_to_simulation_steps(2.49, 1.0) == 2
    assert duration_to_simulation_steps(2.50, 1.0) == 3


def test_empty_map_produces_valid_monotonic_timed_path() -> None:
    grid_map = GridMap(4, 3)

    plan = _plan(grid_map, (0, 1), (3, 1))

    assert plan is not None
    assert plan.spatial_path[0] == (0, 1)
    assert plan.spatial_path[-1] == (3, 1)
    assert plan.timed_states == (
        (0, 1, 0),
        (1, 1, 1),
        (2, 1, 2),
        (3, 1, 3),
    )
    assert plan.actions == ("RIGHT", "RIGHT", "RIGHT")
    assert plan.planned_wait_actions == 0
    assert plan.planned_move_actions == 3
    assert plan.estimated_duration == 3.0
    assert all(
        second[2] == first[2] + 1
        for first, second in zip(plan.timed_states, plan.timed_states[1:])
    )
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(plan.spatial_path, plan.spatial_path[1:])
    )


def test_corridor_crossing_chooses_wait_then_move() -> None:
    grid_map = GridMap(3, 1)

    plan = _plan(
        grid_map,
        (0, 0),
        (2, 0),
        pedestrian_position=(1.0, -1.0),
        pedestrian_velocity=(0.0, 1.0),
        pedestrian_target=(1.0, 2.0),
        collision_distance=0.4,
    )

    assert plan is not None
    assert plan.actions == ("WAIT", "RIGHT", "RIGHT")
    assert plan.spatial_path == ((0, 0), (0, 0), (1, 0), (2, 0))
    assert plan.timed_states == (
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 2),
        (2, 0, 3),
    )
    assert plan.planned_wait_actions == 1
    assert plan.planned_move_actions == 2


def test_wait_is_not_chosen_when_unnecessary() -> None:
    grid_map = GridMap(3, 1)

    plan = _plan(grid_map, (0, 0), (2, 0))

    assert plan is not None
    assert plan.actions == ("RIGHT", "RIGHT")
    assert plan.planned_wait_actions == 0


def test_obstacles_are_respected() -> None:
    grid_map = GridMap(4, 3)
    grid_map.add_obstacle((1, 1))
    grid_map.add_obstacle((2, 1))

    plan = _plan(grid_map, (0, 1), (3, 1))

    assert plan is not None
    assert (1, 1) not in plan.spatial_path
    assert (2, 1) not in plan.spatial_path
    assert all(grid_map.is_free(cell) for cell in plan.spatial_path)
    assert all(
        first == second
        or abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(plan.spatial_path, plan.spatial_path[1:])
    )


def test_repeated_planning_is_deterministic() -> None:
    grid_map = GridMap(4, 3)
    arguments = dict(
        pedestrian_position=(1.5, -1.0),
        pedestrian_velocity=(0.0, 0.75),
        pedestrian_target=(1.5, 2.0),
        collision_distance=0.34,
    )

    plans = [
        _plan(grid_map, (0, 1), (3, 1), **arguments)
        for _ in range(10)
    ]

    assert plans == [plans[0]] * 10


def test_planning_horizon_is_never_exceeded() -> None:
    grid_map = GridMap(4, 1)

    plan = _plan(
        grid_map,
        (0, 0),
        (3, 0),
        max_time_seconds=3.0,
    )
    too_short = _plan(
        grid_map,
        (0, 0),
        (3, 0),
        max_time_seconds=2.99,
    )

    assert plan is not None
    assert max(state[2] for state in plan.timed_states) <= 3
    assert too_short is None


def test_predicted_collision_at_node_is_rejected() -> None:
    grid_map = GridMap(2, 1)

    plan = _plan(
        grid_map,
        (0, 0),
        (1, 0),
        pedestrian_position=(1.0, 0.0),
        pedestrian_velocity=(0.0, 0.0),
        pedestrian_target=(1.0, 0.0),
        max_time_seconds=4.0,
    )

    assert plan is None


def test_predicted_edge_crossing_is_rejected() -> None:
    grid_map = GridMap(2, 1)

    plan = _plan(
        grid_map,
        (0, 0),
        (1, 0),
        pedestrian_position=(0.5, -0.5),
        pedestrian_velocity=(0.0, 1.0),
        pedestrian_target=(0.5, 1.0),
        collision_distance=0.1,
        max_time_seconds=1.0,
    )

    assert plan is None


def test_social_cost_is_evaluated_at_arrival_time() -> None:
    cost_at_start = compute_time_aligned_social_cost(
        (1, 0),
        0,
        pedestrian_position=(1.0, -1.0),
        pedestrian_velocity=(0.0, 1.0),
        pedestrian_target=(1.0, 2.0),
        move_duration=0.5,
        grid_scale=1.0,
        social_distance=2.0,
        social_weight=3.0,
    )
    cost_at_arrival = compute_time_aligned_social_cost(
        (1, 0),
        2,
        pedestrian_position=(1.0, -1.0),
        pedestrian_velocity=(0.0, 1.0),
        pedestrian_target=(1.0, 2.0),
        move_duration=0.5,
        grid_scale=1.0,
        social_distance=2.0,
        social_weight=3.0,
    )

    assert cost_at_start == 3.0
    assert cost_at_arrival == 12.0
