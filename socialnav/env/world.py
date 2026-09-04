"""Deterministic static-navigation demo for SocialNav-Bench."""

import time

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
from socialnav.planners.astar import astar

SIMULATION_STEP = 1.0 / 240.0
STEPS_PER_CELL = 90
ROBOT_RADIUS = 0.18
ROBOT_HEIGHT = 0.20
OBSTACLE_HEIGHT = 0.50


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


def _follow_path(
    robot_id: int, path: list[Coordinate], client_id: int
) -> None:
    height = ROBOT_HEIGHT / 2 + 0.01
    positions = interpolate_path(path, steps_per_cell=STEPS_PER_CELL)

    for x, y in positions:
        if not p.isConnected(client_id):
            return
        p.resetBasePositionAndOrientation(
            robot_id,
            (x, y, height),
            (0.0, 0.0, 0.0, 1.0),
            physicsClientId=client_id,
        )
        p.stepSimulation(physicsClientId=client_id)
        time.sleep(SIMULATION_STEP)


def main() -> None:
    """Run the fixed A* static-navigation demonstration."""
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
        _follow_path(robot_id, path, client_id)

        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(SIMULATION_STEP)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()
