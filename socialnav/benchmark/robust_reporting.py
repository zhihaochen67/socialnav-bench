"""Aggregate reporting for robust space-time execution traces."""

from collections import Counter
from dataclasses import asdict

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


def summarize_shield_execution(
    scenarios: list[Scenario],
    results: list[EpisodeResult],
    traces: list[EpisodeTrace],
) -> dict[str, object]:
    """Aggregate predictive shield decisions and residual failures."""
    predicted_separations = [
        trace.shield_min_predicted_separation
        for trace in traces
        if trace.shield_min_predicted_separation is not None
    ]
    collision_evidence = [
        {
            "scenario_id": scenario.scenario_id,
            "attributions": [
                asdict(attribution)
                for attribution in trace.shield_collision_attributions
            ],
        }
        for scenario, result, trace in zip(scenarios, results, traces)
        if result.human_collision
    ]
    return {
        "shield_checks": sum(trace.shield_checks for trace in traces),
        "shield_activations": sum(
            trace.shield_activations for trace in traces
        ),
        "shield_safe_passthroughs": sum(
            trace.shield_safe_passthroughs for trace in traces
        ),
        "unsafe_planned_moves": sum(
            trace.unsafe_planned_moves for trace in traces
        ),
        "unsafe_waits": sum(trace.unsafe_waits for trace in traces),
        "local_override_count": sum(
            trace.local_override_count for trace in traces
        ),
        "local_override_actions": dict(
            sorted(
                Counter(
                    action
                    for trace in traces
                    for action in trace.local_override_actions
                ).items()
            )
        ),
        "candidate_actions_evaluated": sum(
            trace.candidate_actions_evaluated for trace in traces
        ),
        "candidate_actions_safe": sum(
            trace.candidate_actions_safe for trace in traces
        ),
        "shield_trigger_reasons": dict(
            sorted(
                Counter(
                    reason
                    for trace in traces
                    for reason in trace.shield_trigger_reasons
                ).items()
            )
        ),
        "minimum_predicted_separation": min(
            predicted_separations,
            default=None,
        ),
        "post_override_replans": sum(
            trace.post_override_replans for trace in traces
        ),
        "post_override_replan_successes": sum(
            trace.post_override_replan_successes for trace in traces
        ),
        "post_override_replan_failures": sum(
            trace.post_override_replan_failures for trace in traces
        ),
        "no_safe_local_action_events": sum(
            trace.no_safe_local_action_events for trace in traces
        ),
        "episodes_with_no_safe_local_action": sum(
            trace.no_safe_local_action_events > 0 for trace in traces
        ),
        "stationary_pedestrian_into_robot_collisions": sum(
            any(
                attribution.pedestrian_moved_into_robot
                for attribution in trace.shield_collision_attributions
            )
            for trace in traces
        ),
        "collision_evidence": collision_evidence,
        "remaining_failures": [
            {
                "scenario_id": scenario.scenario_id,
                "reason": trace.robust_episode_failure_reason,
            }
            for scenario, result, trace in zip(scenarios, results, traces)
            if not result.success
        ],
    }
