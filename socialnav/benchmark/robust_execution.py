"""Pure progress-stall detection and failed-replan suppression."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Literal

from socialnav.env.grid_map import Coordinate
from socialnav.metrics import Position
from socialnav.planners.pedestrian_prediction import PREDICTION_EPSILON

from .space_time_diagnostics import REPEATED_STATE_QUANTIZATION

StallEvent = Literal["exact_zero", "progress_stall"]


@dataclass
class ProgressStallDetector:
    """Detect exact-zero or insufficient displacement over one full window."""

    initial_position: Position
    window_steps: int
    robot_speed: float
    simulation_dt: float
    _positions: list[Position] = field(init=False, repr=False)
    _speed_scales: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_steps <= 0:
            raise ValueError("window_steps must be positive")
        if self.robot_speed <= 0.0:
            raise ValueError("robot_speed must be positive")
        if self.simulation_dt <= 0.0:
            raise ValueError("simulation_dt must be positive")
        self._positions = [self.initial_position]
        self._speed_scales = []

    @property
    def minimum_meaningful_displacement(self) -> float:
        """One nominal full-speed physics-step distance."""
        return self.robot_speed * self.simulation_dt

    def reset(self, position: Position) -> None:
        """Start a fresh observation window at the supplied physical pose."""
        self._positions = [position]
        self._speed_scales = []

    def observe(
        self,
        position: Position,
        speed_scale: float,
        *,
        intentional_wait: bool = False,
    ) -> StallEvent | None:
        """Observe one completed physics step and return a stall event."""
        if intentional_wait:
            self.reset(position)
            return None

        self._positions.append(position)
        self._speed_scales.append(speed_scale)
        if len(self._speed_scales) > self.window_steps:
            self._speed_scales.pop(0)
            self._positions.pop(0)
        if len(self._speed_scales) < self.window_steps:
            return None

        displacement = hypot(
            self._positions[-1][0] - self._positions[0][0],
            self._positions[-1][1] - self._positions[0][1],
        )
        if all(scale == 0.0 for scale in self._speed_scales):
            event: StallEvent | None = "exact_zero"
        elif displacement < self.minimum_meaningful_displacement:
            event = "progress_stall"
        else:
            event = None

        if event is not None:
            self.reset(position)
        return event


@dataclass(frozen=True)
class FailedReplanSignature:
    """Quantized state and reason associated with one failed robust replan."""

    mapped_start: Coordinate
    robot_position: tuple[int, int]
    pedestrian_position: tuple[int, int]
    pedestrian_velocity: tuple[int, int]
    pedestrian_at_target: bool
    failure_reason: str

    @property
    def state_key(self) -> tuple[object, ...]:
        return (
            self.mapped_start,
            self.robot_position,
            self.pedestrian_position,
            self.pedestrian_velocity,
            self.pedestrian_at_target,
        )


@dataclass
class FailedReplanSuppressor:
    """Suppress an expensive retry while its relevant state is unchanged."""

    quantization: float = REPEATED_STATE_QUANTIZATION
    last_failed_signature: FailedReplanSignature | None = None

    def __post_init__(self) -> None:
        if self.quantization <= 0.0:
            raise ValueError("quantization must be positive")

    def should_suppress(
        self,
        *,
        mapped_start: Coordinate,
        robot_position: Position,
        pedestrian_position: Position,
        pedestrian_velocity: Position,
        pedestrian_target: Position,
    ) -> bool:
        """Return whether the current state matches the last failed search."""
        if self.last_failed_signature is None:
            return False
        return self.last_failed_signature.state_key == self._state_key(
            mapped_start=mapped_start,
            robot_position=robot_position,
            pedestrian_position=pedestrian_position,
            pedestrian_velocity=pedestrian_velocity,
            pedestrian_target=pedestrian_target,
        )

    def record_failure(
        self,
        *,
        mapped_start: Coordinate,
        robot_position: Position,
        pedestrian_position: Position,
        pedestrian_velocity: Position,
        pedestrian_target: Position,
        failure_reason: str,
    ) -> None:
        """Record the one deterministic failure eligible for suppression."""
        state_key = self._state_key(
            mapped_start=mapped_start,
            robot_position=robot_position,
            pedestrian_position=pedestrian_position,
            pedestrian_velocity=pedestrian_velocity,
            pedestrian_target=pedestrian_target,
        )
        self.last_failed_signature = FailedReplanSignature(
            mapped_start=state_key[0],
            robot_position=state_key[1],
            pedestrian_position=state_key[2],
            pedestrian_velocity=state_key[3],
            pedestrian_at_target=state_key[4],
            failure_reason=failure_reason,
        )

    def record_success(self) -> None:
        """Clear stale failed-search state after a successful plan."""
        self.last_failed_signature = None

    def _state_key(
        self,
        *,
        mapped_start: Coordinate,
        robot_position: Position,
        pedestrian_position: Position,
        pedestrian_velocity: Position,
        pedestrian_target: Position,
    ) -> tuple[
        Coordinate,
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        bool,
    ]:
        return (
            mapped_start,
            self._quantize(robot_position),
            self._quantize(pedestrian_position),
            self._quantize(pedestrian_velocity),
            hypot(
                pedestrian_position[0] - pedestrian_target[0],
                pedestrian_position[1] - pedestrian_target[1],
            )
            <= PREDICTION_EPSILON,
        )

    def _quantize(self, position: Position) -> tuple[int, int]:
        return (
            round(position[0] / self.quantization),
            round(position[1] / self.quantization),
        )
