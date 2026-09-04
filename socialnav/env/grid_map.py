"""A minimal two-dimensional occupancy grid."""

Coordinate = tuple[int, int]


class GridMap:
    """Represent free and obstacle cells on a rectangular grid."""

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Grid map dimensions must be positive.")

        self.width = width
        self.height = height
        self._obstacles: set[Coordinate] = set()

    def is_inside(self, coordinate: Coordinate) -> bool:
        """Return whether a coordinate is inside the map boundaries."""
        x, y = coordinate
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, coordinate: Coordinate) -> bool:
        """Return whether a coordinate is inside the map and has no obstacle."""
        return self.is_inside(coordinate) and coordinate not in self._obstacles

    def add_obstacle(self, coordinate: Coordinate) -> None:
        """Mark a coordinate as occupied by an obstacle."""
        if not self.is_inside(coordinate):
            raise ValueError(f"Obstacle coordinate {coordinate} is outside the map.")

        self._obstacles.add(coordinate)

    def get_obstacles(self) -> set[Coordinate]:
        """Return a copy of the obstacle coordinates."""
        return set(self._obstacles)
