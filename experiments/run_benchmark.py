"""Run the reproducible eleven-method SocialNav benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from math import floor
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socialnav.benchmark import (  # noqa: E402
    REPLAN_STOP_SECONDS,
    REPLAN_STOP_STEPS,
    EpisodeTrace,
    MAX_EPISODE_STEPS,
    MethodSummary,
    SUPPORTED_METHODS,
    aggregate_results,
    generate_diverse_scenarios,
    generate_scenarios,
    run_episode_with_trace,
    summarize_robust_execution,
)
from socialnav.env.world import (  # noqa: E402
    HUMAN_COLLISION_DISTANCE,
    ROBOT_SPEED,
    SIMULATION_STEP,
    SLOW_DISTANCE,
)
from socialnav.evaluation import EpisodeResult  # noqa: E402
from socialnav.planners import (  # noqa: E402
    ESCAPE_SPEED_SCALE,
    PREDICTION_HORIZONS,
    PREDICTION_TEMPORAL_WEIGHTS,
    duration_to_simulation_steps,
)

_METHOD_LABELS = {
    "astar": "A*",
    "dynamic": "Dynamic",
    "social": "Social",
    "social_replan": "Social Replan",
    "social_replan_escape": "Social Replan + Escape",
    "social_replan_recovery": "Social Replan + Recovery",
    "social_predictive": "Predictive Social",
    "social_predictive_replan": "Predictive Social Replan",
    "social_spacetime": "Space-Time Social",
    "social_spacetime_replan": "Space-Time Social Replan",
    "social_spacetime_robust": "Robust Space-Time Social",
}
_SCENARIO_MODES = ("controlled", "diverse")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print_table(summaries: dict[str, MethodSummary]) -> None:
    print(
        "Method                    Success  Collision  SPL    PathLen  Time   "
        "MinHumanDist  SocialViolation"
    )
    print(
        "------------------------  -------  ---------  -----  -------  -----  "
        "------------  ---------------"
    )
    for method in SUPPORTED_METHODS:
        summary = summaries[method]
        print(
            f"{_METHOD_LABELS[method]:<24}  "
            f"{summary.success_rate:>7.3f}  "
            f"{summary.collision_rate:>9.3f}  "
            f"{summary.mean_spl:>5.3f}  "
            f"{summary.mean_path_length:>7.3f}  "
            f"{_format_optional(summary.mean_time_to_goal):>5}  "
            f"{_format_optional(summary.mean_minimum_human_distance):>12}  "
            f"{summary.mean_social_violation_rate:>15.3f}"
        )


def _summarize_replans(traces: list[EpisodeTrace]) -> dict[str, float | int]:
    counts = [trace.replan_count for trace in traces]
    return {
        "total_replans": sum(counts),
        "mean_replan_count": sum(counts) / len(counts),
        "max_replan_count": max(counts),
        "successful_replans": sum(
            trace.successful_replans for trace in traces
        ),
        "failed_replans": sum(trace.failed_replans for trace in traces),
    }


def _summarize_recoveries(
    traces: list[EpisodeTrace],
) -> dict[str, float | int | None]:
    counts = [trace.recovery_count for trace in traces]
    path_lengths = [
        path_length
        for trace in traces
        for path_length in trace.recovery_path_lengths
    ]
    return {
        "total_recoveries": sum(counts),
        "mean_recovery_count": sum(counts) / len(counts),
        "max_recovery_count": max(counts),
        "successful_recoveries": sum(
            trace.successful_recoveries for trace in traces
        ),
        "failed_recoveries": sum(
            trace.failed_recoveries for trace in traces
        ),
        "mean_recovery_path_length": (
            sum(path_lengths) / len(path_lengths)
            if path_lengths
            else None
        ),
        "max_recovery_path_length": max(path_lengths, default=0),
    }


def _summarize_spacetime(
    results: list[EpisodeResult],
    traces: list[EpisodeTrace],
) -> dict[str, float | int]:
    planned_waits = [trace.planned_wait_actions for trace in traces]
    executed_waits = [trace.executed_wait_actions for trace in traces]
    plan_counts = [trace.spacetime_plan_count for trace in traces]
    episodes_using_wait = [count > 0 for count in executed_waits]
    return {
        "total_planned_wait_actions": sum(planned_waits),
        "mean_planned_wait_actions": sum(planned_waits) / len(traces),
        "total_executed_wait_actions": sum(executed_waits),
        "mean_executed_wait_actions": sum(executed_waits) / len(traces),
        "successful_episodes_using_wait": sum(
            result.success and used_wait
            for result, used_wait in zip(results, episodes_using_wait)
        ),
        "failed_episodes_using_wait": sum(
            not result.success and used_wait
            for result, used_wait in zip(results, episodes_using_wait)
        ),
        "total_intentional_wait_steps": sum(
            trace.total_intentional_wait_steps for trace in traces
        ),
        "mean_intentional_wait_seconds": (
            sum(trace.total_intentional_wait_steps for trace in traces)
            * SIMULATION_STEP
            / len(traces)
        ),
        "mean_spacetime_plan_count": sum(plan_counts) / len(traces),
        "max_spacetime_plan_count": max(plan_counts),
        "spacetime_planning_failures": sum(
            trace.spacetime_planning_failures for trace in traces
        ),
        "total_reactive_stopped_steps": sum(
            trace.reactive_stopped_steps for trace in traces
        ),
        "mean_reactive_stopped_steps": sum(
            trace.reactive_stopped_steps for trace in traces
        )
        / len(traces),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic SocialNav headless benchmark."
    )
    parser.add_argument("--episodes", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario-mode",
        choices=_SCENARIO_MODES,
        default="controlled",
    )
    parser.add_argument(
        "--pedestrians",
        "--pedestrian-count",
        dest="pedestrians",
        type=_non_negative_int,
        default=1,
        help="deterministic pedestrian count per scenario (default: 1)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    started_at = perf_counter()
    if args.scenario_mode == "controlled":
        scenarios = generate_scenarios(
            args.episodes,
            args.seed,
            pedestrian_count=args.pedestrians,
        )
    else:
        scenarios = generate_diverse_scenarios(
            args.episodes,
            args.seed,
            pedestrian_count=args.pedestrians,
        )
    spacetime_move_duration = scenarios[0].grid_scale / ROBOT_SPEED
    results_by_method = {method: [] for method in SUPPORTED_METHODS}
    traces_by_method = {method: [] for method in SUPPORTED_METHODS}

    for scenario in scenarios:
        for method in SUPPORTED_METHODS:
            result, trace = run_episode_with_trace(
                scenario, method
            )
            results_by_method[method].append(result)
            traces_by_method[method].append(trace)

    summaries = {
        method: aggregate_results(results)
        for method, results in results_by_method.items()
    }
    print(f"Scenario mode: {args.scenario_mode}")
    _print_table(summaries)

    output_name = (
        f"benchmark_seed{args.seed}.json"
        if args.scenario_mode == "controlled"
        else f"benchmark_diverse_seed{args.seed}.json"
    )
    output_path = PROJECT_ROOT / "outputs" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "episodes": args.episodes,
        "scenario_mode": args.scenario_mode,
        "pedestrian_count": args.pedestrians,
        "configuration": {
            "simulation_dt": SIMULATION_STEP,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "max_episode_seconds": MAX_EPISODE_STEPS * SIMULATION_STEP,
            "replan_stop_seconds": REPLAN_STOP_SECONDS,
            "replan_stop_steps": REPLAN_STOP_STEPS,
            "escape_speed_scale": ESCAPE_SPEED_SCALE,
            "recovery_target_clearance": SLOW_DISTANCE,
            "prediction_horizons": list(PREDICTION_HORIZONS),
            "prediction_temporal_weights": list(PREDICTION_TEMPORAL_WEIGHTS),
            "robot_speed": ROBOT_SPEED,
            "human_collision_distance": HUMAN_COLLISION_DISTANCE,
            "spacetime_move_duration_rule": "grid_scale / robot_speed",
            "spacetime_move_duration_seconds": spacetime_move_duration,
            "spacetime_wait_duration_seconds": spacetime_move_duration,
            "spacetime_action_steps": duration_to_simulation_steps(
                spacetime_move_duration,
                SIMULATION_STEP,
            ),
            "spacetime_action_step_rounding": (
                "nearest integer; exact ties round upward"
            ),
            "spacetime_max_time_index": floor(
                MAX_EPISODE_STEPS
                * SIMULATION_STEP
                / spacetime_move_duration
            ),
            "robust_bridge_candidate_rule": (
                "nearest free mapped cell plus its four-connected local cells"
            ),
            "robust_bridge_duration_rule": (
                "euclidean distance / robot_speed"
            ),
            "robust_bridge_tie_breaking": [
                "safe",
                "smaller_bridge_distance",
                "greater_minimum_human_separation",
                "lower_x",
                "lower_y",
            ],
            "robust_prediction_time_rule": (
                "bridge_duration + grid_time_index * move_duration"
            ),
            "robust_egress_tolerance": 1e-12,
            "robust_stall_window_steps": REPLAN_STOP_STEPS,
            "robust_progress_threshold_rule": (
                "robot_speed * simulation_dt"
            ),
            "robust_progress_threshold_meters": (
                ROBOT_SPEED * SIMULATION_STEP
            ),
            "robust_stall_comparison": "displacement < threshold",
            "robust_duplicate_state_quantization": 0.001,
            "methods": {
                "astar": "ordinary A* without reactive avoidance",
                "dynamic": "ordinary A* with reactive avoidance",
                "social": "social A* with reactive avoidance",
                "social_replan": (
                    "social A* with reactive avoidance and online replanning"
                ),
                "social_replan_escape": (
                    "social A* with online replanning and direction-aware "
                    "reactive escape control"
                ),
                "social_replan_recovery": (
                    "social A* with online replanning, direction-aware "
                    "escape control, and local clearance recovery"
                ),
                "social_predictive": (
                    "predictive social A* with reactive avoidance"
                ),
                "social_predictive_replan": (
                    "predictive social A* with reactive avoidance and "
                    "online replanning using current pedestrian velocity"
                ),
                "social_spacetime": (
                    "time-expanded social A* with explicit MOVE/WAIT actions "
                    "and reactive movement safety"
                ),
                "social_spacetime_replan": (
                    "time-expanded social A* with explicit MOVE/WAIT actions, "
                    "reactive movement safety, and sustained-stop replanning"
                ),
                "social_spacetime_robust": (
                    "continuous-start time-expanded social A* with safe "
                    "collision egress, progress-aware replanning, duplicate "
                    "suppression, and direction-aware execution"
                ),
            },
        },
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "methods": {
            method: {
                "summary": asdict(summaries[method]),
                "replanning": _summarize_replans(
                    traces_by_method[method]
                ),
                "recovery": _summarize_recoveries(
                    traces_by_method[method]
                ),
                "spacetime": _summarize_spacetime(
                    results_by_method[method],
                    traces_by_method[method],
                ),
                "robust": (
                    summarize_robust_execution(
                        scenarios,
                        results_by_method[method],
                        traces_by_method[method],
                    )
                    if method == "social_spacetime_robust"
                    else None
                ),
                "episodes": [
                    {
                        "scenario_id": scenario.scenario_id,
                        "result": asdict(result),
                        "trace": {
                            "replan_count": trace.replan_count,
                            "replan_steps": list(trace.replan_steps),
                            "successful_replans": trace.successful_replans,
                            "failed_replans": trace.failed_replans,
                            "recovery_count": trace.recovery_count,
                            "recovery_trigger_steps": list(
                                trace.recovery_trigger_steps
                            ),
                            "successful_recoveries": (
                                trace.successful_recoveries
                            ),
                            "failed_recoveries": (
                                trace.failed_recoveries
                            ),
                            "recovery_path_lengths": list(
                                trace.recovery_path_lengths
                            ),
                            "planned_wait_actions": (
                                trace.planned_wait_actions
                            ),
                            "executed_wait_actions": (
                                trace.executed_wait_actions
                            ),
                            "planned_move_actions": (
                                trace.planned_move_actions
                            ),
                            "spacetime_plan_count": (
                                trace.spacetime_plan_count
                            ),
                            "spacetime_planning_failures": (
                                trace.spacetime_planning_failures
                            ),
                            "total_intentional_wait_steps": (
                                trace.total_intentional_wait_steps
                            ),
                            "reactive_stopped_steps": (
                                trace.reactive_stopped_steps
                            ),
                            "continuous_bridge_attempts": trace.continuous_bridge_attempts,
                            "continuous_bridge_successes": trace.continuous_bridge_successes,
                            "continuous_bridge_failures": trace.continuous_bridge_failures,
                            "bridge_target_cells": list(trace.bridge_target_cells),
                            "bridge_distances": list(trace.bridge_distances),
                            "bridge_min_predicted_separations": list(trace.bridge_min_predicted_separations),
                            "collision_egress_attempts": trace.collision_egress_attempts,
                            "collision_egress_successes": trace.collision_egress_successes,
                            "collision_egress_failures": trace.collision_egress_failures,
                            "progress_stall_events": trace.progress_stall_events,
                            "exact_zero_stall_events": trace.exact_zero_stall_events,
                            "suppressed_duplicate_replans": trace.suppressed_duplicate_replans,
                            "robust_replan_count": trace.robust_replan_count,
                            "robust_replan_successes": trace.robust_replan_successes,
                            "robust_replan_failures": trace.robust_replan_failures,
                            "robust_planning_failure_reasons": list(trace.robust_planning_failure_reasons),
                            "robust_episode_failure_reason": trace.robust_episode_failure_reason,
                            "pedestrian_count": trace.pedestrian_count,
                            "initial_pedestrian_positions": [
                                list(position)
                                for position in trace.initial_pedestrian_positions
                            ],
                            "initial_pedestrian_velocities": [
                                list(velocity)
                                for velocity in trace.initial_pedestrian_velocities
                            ],
                            "pedestrian_targets": [
                                list(target)
                                for target in trace.pedestrian_targets
                            ],
                            "final_pedestrian_positions": [
                                list(position)
                                for position in trace.final_pedestrian_positions
                            ],
                            "per_human_minimum_distances": list(
                                trace.per_human_minimum_distances
                            ),
                            "collision_human_indices": list(
                                trace.collision_human_indices
                            ),
                            "blocking_human_indices": list(
                                trace.blocking_human_indices
                            ),
                            "minimum_predicted_separation": (
                                trace.minimum_predicted_separation
                            ),
                        },
                    }
                    for scenario, result, trace in zip(
                        scenarios,
                        results_by_method[method],
                        traces_by_method[method],
                    )
                ],
            }
            for method in SUPPORTED_METHODS
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    elapsed = perf_counter() - started_at
    print(f"\nSaved: {output_path}")
    print(f"Runtime: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
