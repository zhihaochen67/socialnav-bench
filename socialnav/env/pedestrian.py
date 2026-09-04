"""Deterministic two-dimensional pedestrian motion."""

from math import hypot

Position = tuple[float, float]


class Pedestrian:
    """Move at a constant speed along a straight line to a fixed target."""

    def __init__(
        self,
        start_position: Position,
        target_position: Position,
        speed: float,
    ) -> None:
        if speed < 0:
            raise ValueError("speed must be non-negative")

        self.start_position = start_position
        self.target_position = target_position
        self.speed = speed
        self.position = start_position

    @property
    def has_reached_target(self) -> bool:
        """Return whether the pedestrian is at its target."""
        return self.position == self.target_position

    def advance(self, time_step: float) -> Position:
        """Advance by the supplied number of seconds and return the position."""
        if time_step < 0:
            raise ValueError("time_step must be non-negative")
        if self.has_reached_target or self.speed == 0:
            return self.position

        delta_x = self.target_position[0] - self.position[0]
        delta_y = self.target_position[1] - self.position[1]
        distance_remaining = hypot(delta_x, delta_y)
        movement_distance = self.speed * time_step

        if movement_distance >= distance_remaining:
            self.position = self.target_position
            return self.position

        fraction = movement_distance / distance_remaining
        self.position = (
            self.position[0] + delta_x * fraction,
            self.position[1] + delta_y * fraction,
        )
        return self.position
