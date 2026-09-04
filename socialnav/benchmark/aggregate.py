"""Pure aggregation of episode-level benchmark results."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from socialnav.evaluation import EpisodeResult


@dataclass(frozen=True)
class MethodSummary:
    """Aggregate metrics for one navigation method."""

    episodes: int
    success_rate: float
    collision_rate: float
    mean_path_length: float
    mean_time_to_goal: float | None
    mean_spl: float
    mean_minimum_human_distance: float | None
    mean_social_violation_rate: float


def aggregate_results(results: Sequence[EpisodeResult]) -> MethodSummary:
    """Aggregate one method's episode results."""
    if not results:
        raise ValueError("results must not be empty")

    successful_times = [
        result.time_to_goal
        for result in results
        if result.time_to_goal is not None
    ]
    defined_minimum_distances = [
        result.minimum_human_distance
        for result in results
        if result.minimum_human_distance is not None
    ]

    return MethodSummary(
        episodes=len(results),
        success_rate=fmean(result.success for result in results),
        collision_rate=fmean(
            result.human_collision or result.obstacle_collision
            for result in results
        ),
        mean_path_length=fmean(result.path_length for result in results),
        mean_time_to_goal=(
            fmean(successful_times) if successful_times else None
        ),
        mean_spl=fmean(result.spl for result in results),
        mean_minimum_human_distance=(
            fmean(defined_minimum_distances)
            if defined_minimum_distances
            else None
        ),
        mean_social_violation_rate=fmean(
            result.social_violation_rate for result in results
        ),
    )
