"""Aggregate reporting for robust space-time execution traces."""

from collections import Counter

from socialnav.evaluation import EpisodeResult

from .diagnostics import EpisodeTrace
from .scenario import Scenario


def summarize_robust_execution(
    scenarios: list[Scenario],
    results: list[EpisodeResult],
    traces: list[EpisodeTrace],
) -> dict[str, object]:
    """Summarize bridge, egress, stall, suppression, and residual failures."""
    bridge_distances = [
        distance
        for trace in traces
        for distance in trace.bridge_distances
    ]
    minimum_separations = [
        separation
        for trace in traces
        for separation in trace.bridge_min_predicted_separations
    ]
    return {
        "continuous_bridge_attempts": sum(
            trace.continuous_bridge_attempts for trace in traces
        ),
        "continuous_bridge_successes": sum(
            trace.continuous_bridge_successes for trace in traces
        ),
        "continuous_bridge_failures": sum(
            trace.continuous_bridge_failures for trace in traces
        ),
        "mean_bridge_distance": (
            sum(bridge_distances) / len(bridge_distances)
            if bridge_distances
            else None
        ),
        "minimum_bridge_predicted_separation": min(
            minimum_separations,
            default=None,
        ),
        "collision_egress_attempts": sum(
            trace.collision_egress_attempts for trace in traces
        ),
        "collision_egress_successes": sum(
            trace.collision_egress_successes for trace in traces
        ),
        "collision_egress_failures": sum(
            trace.collision_egress_failures for trace in traces
        ),
        "progress_stall_events": sum(
            trace.progress_stall_events for trace in traces
        ),
        "exact_zero_stall_events": sum(
            trace.exact_zero_stall_events for trace in traces
        ),
        "suppressed_duplicate_replans": sum(
            trace.suppressed_duplicate_replans for trace in traces
        ),
        "robust_replan_count": sum(
            trace.robust_replan_count for trace in traces
        ),
        "robust_replan_successes": sum(
            trace.robust_replan_successes for trace in traces
        ),
        "robust_replan_failures": sum(
            trace.robust_replan_failures for trace in traces
        ),
        "planning_failure_reasons": dict(
            sorted(
                Counter(
                    reason
                    for trace in traces
                    for reason in trace.robust_planning_failure_reasons
                ).items()
            )
        ),
        "remaining_failures": [
            {
                "scenario_id": scenario.scenario_id,
                "reason": trace.robust_episode_failure_reason,
            }
            for scenario, result, trace in zip(scenarios, results, traces)
            if not result.success
        ],
    }
