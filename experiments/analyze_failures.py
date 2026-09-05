"""Analyze failed social-planner benchmark episodes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from math import ceil
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socialnav.benchmark import (  # noqa: E402
    LATE_EPISODE_FRACTION,
    LATE_STOPPED_FRACTION_THRESHOLD,
    LONGEST_STOPPED_FRACTION_THRESHOLD,
    MAX_EPISODE_STEPS,
    PATH_NEAR_THRESHOLD,
    STOPPED_SPEED_TOLERANCE,
    FailureDiagnostic,
    diagnose_failure,
    generate_diverse_scenarios,
    generate_scenarios,
    run_episode_with_trace,
)
from socialnav.env.world import (  # noqa: E402
    SIMULATION_STEP,
    SLOW_DISTANCE,
    STOP_DISTANCE,
)
from socialnav.planners import CLEARANCE_EPSILON  # noqa: E402

_SCENARIO_MODES = ("controlled", "diverse")
_DIAGNOSTIC_METHODS = (
    "social",
    "social_replan",
    "social_replan_escape",
    "social_replan_recovery",
)
_FAILURE_REASONS = (
    "pedestrian_blocking_path",
    "reactive_wait_timeout",
    "collision",
    "goal_not_reached",
    "other",
)
_CLASSIFICATION_PRECEDENCE = (
    "collision",
    "pedestrian_blocking_path",
    "reactive_wait_timeout",
    "goal_not_reached",
    "other",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose failed social-method benchmark episodes."
    )
    parser.add_argument("--episodes", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method",
        choices=_DIAGNOSTIC_METHODS,
        default="social",
    )
    parser.add_argument(
        "--scenario-mode",
        choices=_SCENARIO_MODES,
        default="diverse",
    )
    return parser


def _representative_scenarios(
    diagnostics: list[FailureDiagnostic],
) -> list[dict[str, str]]:
    representatives = []
    for reason in _FAILURE_REASONS:
        match = next(
            (
                diagnostic
                for diagnostic in diagnostics
                if diagnostic.likely_failure_reason == reason
            ),
            None,
        )
        if match is not None:
            representatives.append(
                {
                    "scenario_id": match.scenario_id,
                    "reason": match.likely_failure_reason,
                }
            )
    return representatives


def _summarize(
    episodes: int,
    diagnostics: list[FailureDiagnostic],
) -> dict[str, object]:
    reason_counts = Counter(
        diagnostic.likely_failure_reason for diagnostic in diagnostics
    )
    failures = len(diagnostics)
    mean_stopped_fraction = (
        sum(
            diagnostic.robot_stopped_fraction
            for diagnostic in diagnostics
        )
        / failures
        if failures
        else None
    )
    wait_timeout_failures = [
        diagnostic
        for diagnostic in diagnostics
        if (
            diagnostic.timed_out
            and diagnostic.late_robot_stopped_fraction
            >= LATE_STOPPED_FRACTION_THRESHOLD
            and diagnostic.longest_consecutive_stop_steps
            >= ceil(
                diagnostic.steps * LONGEST_STOPPED_FRACTION_THRESHOLD
            )
        )
    ]
    pedestrian_blocking_evidence = [
        diagnostic
        for diagnostic in wait_timeout_failures
        if diagnostic.pedestrian_final_position_on_or_near_path
        and diagnostic.final_robot_to_pedestrian_distance
        <= STOP_DISTANCE + STOPPED_SPEED_TOLERANCE
    ]

    return {
        "episodes": episodes,
        "successes": episodes - failures,
        "failures": failures,
        "timeouts": sum(
            diagnostic.timed_out for diagnostic in diagnostics
        ),
        "wait_timeout_failures": len(wait_timeout_failures),
        "pedestrian_blocking_evidence": len(
            pedestrian_blocking_evidence
        ),
        "counts_by_reason": {
            reason: reason_counts[reason] for reason in _FAILURE_REASONS
        },
        "pedestrian_final_position_on_or_near_route": sum(
            diagnostic.pedestrian_final_position_on_or_near_path
            for diagnostic in diagnostics
        ),
        "pedestrian_target_on_or_near_route": sum(
            diagnostic.pedestrian_target_on_path
            for diagnostic in diagnostics
        ),
        "mean_stopped_fraction": mean_stopped_fraction,
        "mean_replan_count_among_failures": (
            sum(diagnostic.replan_count for diagnostic in diagnostics)
            / failures
            if failures
            else None
        ),
        "max_replan_count_among_failures": max(
            (diagnostic.replan_count for diagnostic in diagnostics),
            default=0,
        ),
        "mean_recovery_count_among_failures": (
            sum(diagnostic.recovery_count for diagnostic in diagnostics)
            / failures
            if failures
            else None
        ),
        "max_recovery_count_among_failures": max(
            (diagnostic.recovery_count for diagnostic in diagnostics),
            default=0,
        ),
        "successful_recoveries": sum(
            diagnostic.successful_recoveries for diagnostic in diagnostics
        ),
        "failed_recoveries": sum(
            diagnostic.failed_recoveries for diagnostic in diagnostics
        ),
        "failed_replanning_calls": sum(
            diagnostic.failed_replans for diagnostic in diagnostics
        ),
        "human_collision_failures": sum(
            diagnostic.human_collision for diagnostic in diagnostics
        ),
        "obstacle_collision_failures": sum(
            diagnostic.obstacle_collision for diagnostic in diagnostics
        ),
        "planning_failures": 0,
    }


def _print_summary(
    method: str,
    summary: dict[str, object],
    representatives: list[dict[str, str]],
) -> None:
    counts = summary["counts_by_reason"]
    assert isinstance(counts, dict)
    mean_stopped = summary["mean_stopped_fraction"]
    mean_stopped_text = (
        "-"
        if mean_stopped is None
        else f"{mean_stopped:.3f}"
    )

    if method == "social_replan_recovery":
        method_label = "Social Replan + Recovery"
    elif method == "social_replan_escape":
        method_label = "Social Replan + Escape"
    elif method == "social_replan":
        method_label = "Social Replan"
    else:
        method_label = "Social"
    title = f"{method_label} Failure Analysis"
    print(title)
    print("-" * len(title))
    print(f"Episodes: {summary['episodes']}")
    print(f"Failures: {summary['failures']}")
    print(f"Timeouts: {summary['timeouts']}")
    print(f"Wait timeout failures: {summary['wait_timeout_failures']}")
    print(
        "Pedestrian blocking evidence: "
        f"{summary['pedestrian_blocking_evidence']}"
    )
    print(
        "Pedestrian final position on/near route: "
        f"{summary['pedestrian_final_position_on_or_near_route']}"
    )
    print(
        "Mean stopped fraction among failures: "
        f"{mean_stopped_text}"
    )
    mean_replans = summary["mean_replan_count_among_failures"]
    print(
        "Mean replan count among failures: "
        f"{'-' if mean_replans is None else f'{mean_replans:.3f}'}"
    )
    print(
        "Max replan count among failures: "
        f"{summary['max_replan_count_among_failures']}"
    )
    print(
        f"Failed replanning calls: {summary['failed_replanning_calls']}"
    )
    mean_recoveries = summary["mean_recovery_count_among_failures"]
    print(
        "Mean recovery count among failures: "
        f"{'-' if mean_recoveries is None else f'{mean_recoveries:.3f}'}"
    )
    print(
        "Max recovery count among failures: "
        f"{summary['max_recovery_count_among_failures']}"
    )
    print(f"Successful recoveries: {summary['successful_recoveries']}")
    print(f"Failed recoveries: {summary['failed_recoveries']}")
    print(
        "Pedestrian blocking path (classified): "
        f"{counts['pedestrian_blocking_path']}"
    )
    print(f"Reactive wait timeout: {counts['reactive_wait_timeout']}")
    print(f"Collision failures: {counts['collision']}")
    print(f"Goal not reached: {counts['goal_not_reached']}")
    print(f"Other: {counts['other']}")
    print(f"Planning failures: {summary['planning_failures']}")
    print(
        "Obstacle collision failures: "
        f"{summary['obstacle_collision_failures']}"
    )
    print("\nRepresentative scenarios:")
    if representatives:
        for representative in representatives:
            print(
                f"- {representative['scenario_id']}: "
                f"{representative['reason']}"
            )
    else:
        print("- none")


def main() -> None:
    args = build_parser().parse_args()
    started_at = perf_counter()

    if args.scenario_mode == "controlled":
        scenarios = generate_scenarios(args.episodes, args.seed)
    else:
        scenarios = generate_diverse_scenarios(args.episodes, args.seed)

    diagnostics = []
    for scenario in scenarios:
        result, trace = run_episode_with_trace(scenario, args.method)
        diagnostic = diagnose_failure(
            scenario,
            args.method,
            result,
            trace,
        )
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    summary = _summarize(args.episodes, diagnostics)
    representatives = _representative_scenarios(diagnostics)
    payload = {
        "schema_version": 1,
        "configuration": {
            "episodes": args.episodes,
            "seed": args.seed,
            "scenario_mode": args.scenario_mode,
            "method": args.method,
            "simulation_dt": SIMULATION_STEP,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "max_episode_seconds": MAX_EPISODE_STEPS * SIMULATION_STEP,
            "path_near_threshold_metres": PATH_NEAR_THRESHOLD,
            "reactive_stop_distance_metres": STOP_DISTANCE,
            "recovery_target_clearance_metres": SLOW_DISTANCE,
            "recovery_clearance_epsilon": CLEARANCE_EPSILON,
            "stopped_speed_scale_tolerance": STOPPED_SPEED_TOLERANCE,
            "late_episode_fraction": LATE_EPISODE_FRACTION,
            "late_stopped_fraction_threshold": (
                LATE_STOPPED_FRACTION_THRESHOLD
            ),
            "longest_stopped_fraction_threshold": (
                LONGEST_STOPPED_FRACTION_THRESHOLD
            ),
            "classification_precedence": list(
                _CLASSIFICATION_PRECEDENCE
            ),
            "classification_rules": {
                "collision": "failed and any collision flag",
                "pedestrian_blocking_path": (
                    "failed, no collision, timed out, wait pattern, "
                    "pedestrian final position within route threshold, "
                    "and final robot-pedestrian distance within stop distance"
                ),
                "reactive_wait_timeout": (
                    "failed, no collision, timed out, and wait pattern"
                ),
                "goal_not_reached": (
                    "failed, no collision, and did not time out"
                ),
                "other": (
                    "failed timeout without collision or wait pattern"
                ),
            },
        },
        "summary": summary,
        "failure_diagnostics": [
            asdict(diagnostic) for diagnostic in diagnostics
        ],
        "representative_scenarios": representatives,
    }

    output_name = (
        f"failure_analysis_{args.scenario_mode}_seed{args.seed}.json"
        if args.method == "social"
        else (
            f"failure_analysis_{args.method}_{args.scenario_mode}_"
            f"seed{args.seed}.json"
        )
    )
    output_path = (
        PROJECT_ROOT
        / "outputs"
        / output_name
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    _print_summary(args.method, summary, representatives)
    elapsed = perf_counter() - started_at
    print(f"\nSaved: {output_path}")
    print(f"Runtime: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
