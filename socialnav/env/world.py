"""Minimal PyBullet world for SocialNav-Bench."""

import time

import pybullet as p
import pybullet_data


def main() -> None:
    """Open a simple PyBullet world and run it until it is closed."""
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Could not connect to PyBullet in GUI mode.")

    try:
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=client_id
        )
        p.setGravity(0, 0, -9.81, physicsClientId=client_id)
        p.loadURDF("plane.urdf", physicsClientId=client_id)

        radius = 0.2
        height = 0.15
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=client_id,
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=(0.2, 0.5, 0.9, 1.0),
            physicsClientId=client_id,
        )
        p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=(0.0, 0.0, height / 2 + 0.01),
            physicsClientId=client_id,
        )

        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()
