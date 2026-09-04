"""Static A* navigation with reactive avoidance and personal-space display."""

import time
from math import cos, sin, tau

import pybullet as p
import pybullet_data

from socialnav.env.demo_map import (
    CELL_SIZE,
    GOAL,
    GRID_HEIGHT,
    GRID_WIDTH,
    START,
    build_demo_grid,
    grid_to_world,
    interpolate_path,
)
from socialnav.env.grid_map import Coordinate, GridMap
from socialnav.env.pedestrian import Pedestrian
from socialnav.planners.astar import astar
from socialnav.planners.dynamic_avoidance import compute_speed_scale

SIMULATION_STEP = 1.0 / 240.0
STEPS_PER_CELL = 90
ROBOT_RADIUS = 0.18
ROBOT_HEIGHT = 0.20
OBSTACLE_HEIGHT = 0.50
PEDESTRIAN_RADIUS = 0.16
PEDESTRIAN_HEIGHT = 0.80
PEDESTRIAN_START = (CELL_SIZE, CELL_SIZE * (GRID_HEIGHT - 1))
PEDESTRIAN_TARGET = (CELL_SIZE, CELL_SIZE)
PEDESTRIAN_SPEED = 0.40
STOP_DISTANCE = 0.55
SLOW_DISTANCE = 1.25
SOCIAL_DISTANCE = 1.0
PERSONAL_SPACE_SEGMENTS = 32


def _configure_world(client_id: int) -> None:
    p.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setTimeStep(SIMULATION_STEP, physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    camera_target = (
        CELL_SIZE * (GRID_WIDTH - 1) / 2,
        CELL_SIZE * (GRID_HEIGHT - 1) / 2,
        0.0,
    )
    p.resetDebugVisualizerCamera(
        cameraDistance=6.2,
        cameraYaw=35,
        cameraPitch=-65,
        cameraTargetPosition=camera_target,
        physicsClientId=client_id,
    )


def _render_obstacles(grid_map: GridMap, client_id: int) -> None:
    half_width = CELL_SIZE * 0.38
    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=(half_width, half_width, OBSTACLE_HEIGHT / 2),
        physicsClientId=client_id,
    )
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=(half_width, half_width, OBSTACLE_HEIGHT / 2),
        rgbaColor=(0.65, 0.18, 0.15, 1.0),
        physicsClientId=client_id,
    )

    for coordinate in sorted(grid_map.get_obstacles()):
        x, y = grid_to_world(coordinate)
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=(x, y, OBSTACLE_HEIGHT / 2),
            physicsClientId=client_id,
        )


def _render_goal(client_id: int) -> None:
    marker_height = 0.025
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=ROBOT_RADIUS * 1.5,
        length=marker_height,
        rgbaColor=(0.15, 0.8, 0.25, 1.0),
        physicsClientId=client_id,
    )
    x, y = grid_to_world(GOAL)
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=(x, y, marker_height / 2 + 0.005),
        physicsClientId=client_id,
    )


def _render_path(path: list[Coordinate], client_id: int) -> None:
    marker_shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.045,
        rgbaColor=(1.0, 0.65, 0.05, 1.0),
        physicsClientId=client_id,
    )
    world_path = [grid_to_world(coordinate) for coordinate in path]

    for x, y in world_path[1:-1]:
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=marker_shape,
            basePosition=(x, y, 0.04),
            physicsClientId=client_id,
        )

    for first, second in zip(world_path, world_path[1:]):
        p.addUserDebugLine(
            (first[0], first[1], 0.035),
            (second[0], second[1], 0.035),
            lineColorRGB=(1.0, 0.65, 0.05),
            lineWidth=3.0,
            physicsClientId=client_id,
        )


def _create_robot(client_id: int) -> int:
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=ROBOT_RADIUS,
        height=ROBOT_HEIGHT,
        physicsClientId=client_id,
    )
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=ROBOT_RADIUS,
        length=ROBOT_HEIGHT,
        rgbaColor=(0.2, 0.5, 0.9, 1.0),
        physicsClientId=client_id,
    )
    x, y = grid_to_world(START)
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=(x, y, ROBOT_HEIGHT / 2 + 0.01),
        physicsClientId=client_id,
    )


def _create_pedestrian(pedestrian: Pedestrian, client_id: int) -> int:
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=PEDESTRIAN_RADIUS,
        height=PEDESTRIAN_HEIGHT,
        physicsClientId=client_id,
    )
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=PEDESTRIAN_RADIUS,
        length=PEDESTRIAN_HEIGHT,
        rgbaColor=(0.75, 0.2, 0.85, 1.0),
        physicsClientId=client_id,
    )
    x, y = pedestrian.position
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=(x, y, PEDESTRIAN_HEIGHT / 2 + 0.01),
        physicsClientId=client_id,
    )


def _render_personal_space(
    pedestrian_id: int,
    client_id: int,
    radius: float = SOCIAL_DISTANCE,
    segments: int = PERSONAL_SPACE_SEGMENTS,
) -> None:
    """Draw a ground ring parented to the moving pedestrian."""
    line_height = -PEDESTRIAN_HEIGHT / 2

    for segment in range(segments):
        start_angle = tau * segment / segments
        end_angle = tau * (segment + 1) / segments
        p.addUserDebugLine(
            (radius * cos(start_angle), radius * sin(start_angle), line_height),
            (radius * cos(end_angle), radius * sin(end_angle), line_height),
            lineColorRGB=(0.1, 0.8, 0.85),
            lineWidth=2.0,
            parentObjectUniqueId=pedestrian_id,
            parentLinkIndex=-1,
            physicsClientId=client_id,
        )


def _advance_pedestrian(
    pedestrian: Pedestrian, pedestrian_id: int, client_id: int
) -> None:
    x, y = pedestrian.advance(SIMULATION_STEP)
    p.resetBasePositionAndOrientation(
        pedestrian_id,
        (x, y, PEDESTRIAN_HEIGHT / 2 + 0.01),
        (0.0, 0.0, 0.0, 1.0),
        physicsClientId=client_id,
    )


def _follow_path(
    robot_id: int,
    path: list[Coordinate],
    pedestrian: Pedestrian,
    pedestrian_id: int,
    client_id: int,
    stop_distance: float = STOP_DISTANCE,
    slow_distance: float = SLOW_DISTANCE,
) -> None:
    height = ROBOT_HEIGHT / 2 + 0.01
    positions = interpolate_path(path, steps_per_cell=STEPS_PER_CELL)
    if not positions:
        return

    last_position_index = len(positions) - 1
    path_progress = 0.0
    robot_position = positions[0]

    while path_progress < last_position_index:
        if not p.isConnected(client_id):
            return

        speed_scale = compute_speed_scale(
            robot_position,
            pedestrian.position,
            stop_distance,
            slow_distance,
        )
        path_progress = min(
            path_progress + speed_scale,
            float(last_position_index),
        )

        lower_index = int(path_progress)
        if lower_index == last_position_index:
            robot_position = positions[-1]
        else:
            fraction = path_progress - lower_index
            start_x, start_y = positions[lower_index]
            end_x, end_y = positions[lower_index + 1]
            robot_position = (
                start_x + (end_x - start_x) * fraction,
                start_y + (end_y - start_y) * fraction,
            )

        p.resetBasePositionAndOrientation(
            robot_id,
            (*robot_position, height),
            (0.0, 0.0, 0.0, 1.0),
            physicsClientId=client_id,
        )
        _advance_pedestrian(pedestrian, pedestrian_id, client_id)
        p.stepSimulation(physicsClientId=client_id)
        time.sleep(SIMULATION_STEP)


def main() -> None:
    """Run static A* with local reactive pedestrian avoidance."""
    grid_map = build_demo_grid()
    path = astar(grid_map, START, GOAL)
    if path is None:
        raise RuntimeError(
            "A* could not find a path for the static navigation demo."
        )

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Could not connect to PyBullet in GUI mode.")

    try:
        _configure_world(client_id)
        _render_obstacles(grid_map, client_id)
        _render_goal(client_id)
        _render_path(path, client_id)
        robot_id = _create_robot(client_id)
        pedestrian = Pedestrian(
            start_position=PEDESTRIAN_START,
            target_position=PEDESTRIAN_TARGET,
            speed=PEDESTRIAN_SPEED,
        )
        pedestrian_id = _create_pedestrian(pedestrian, client_id)
        _render_personal_space(pedestrian_id, client_id)
        _follow_path(
            robot_id,
            path,
            pedestrian,
            pedestrian_id,
            client_id,
        )

        while p.isConnected(client_id):
            _advance_pedestrian(pedestrian, pedestrian_id, client_id)
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(SIMULATION_STEP)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()
