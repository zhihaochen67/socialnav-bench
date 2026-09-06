"""Evidence-only diagnosis of residual space-time navigation failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from socialnav.benchmark import (  # noqa: E402
    MAX_EPISODE_STEPS,
    STOPPED_SPEED_TOLERANCE,
    diagnose_failure,
    generate_diverse_scenarios,
    generate_scenarios,
    run_episode_with_trace,
)
from socialnav.benchmark.space_time_diagnostics import (  # noqa: E402
    REPEATED_STATE_QUANTIZATION,
    SpaceTimePlanningCall,
    group_repeated_failed_planning_calls,
)
from socialnav.env.demo_map import SOCIAL_DISTANCE, SOCIAL_WEIGHT  # noqa: E402
from socialnav.env.world import (  # noqa: E402
    HUMAN_COLLISION_DISTANCE,
    ROBOT_SPEED,
    SIMULATION_STEP,
)
from socialnav.planners.space_time_planner import (  # noqa: E402
    SpaceTimeFailureReason,
)

_SCENARIO_MODES = ("controlled", "diverse")
_SPACE_TIME_METHODS = ("social_spacetime", "social_spacetime_replan")
_FAILURE_CATEGORIES: tuple[SpaceTimeFailureReason, ...] = (
    "invalid_start",
    "invalid_goal",
    "start_in_predicted_collision",
    "no_safe_first_action",
    "goal_unreachable_static",
    "time_horizon_exhausted",
    "search_exhausted",
    "other",
)
_RESIDUAL_EPISODE_IDS = (
    "diverse-seed-42-episode-0021",
    "diverse-seed-42-episode-0030",
    "diverse-seed-42-episode-0044",
    "diverse-seed-42-episode-0058",
    "diverse-seed-42-episode-0069",
    "diverse-seed-42-episode-0075",
    "diverse-seed-42-episode-0084",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose space-time planner failures without changing behavior."
    )
    parser.add_argument("--episodes", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario-mode",
        choices=_SCENARIO_MODES,
        default="diverse",
    )
    parser.add_argument(
        "--method",
        choices=_SPACE_TIME_METHODS,
        default="social_spacetime_replan",
    )
    return parser


def _failed_replans(
    calls: list[SpaceTimePlanningCall],
) -> list[SpaceTimePlanningCall]:
    return [
        call
        for call in calls
        if not call.is_initial_plan and call.failure_reason is not None
    ]


def _failure_state_signature(
    call: SpaceTimePlanningCall,
) -> tuple[object, ...]:
    def quantize(position: tuple[float, float]) -> tuple[int, int]:
        return (
            round(position[0] / REPEATED_STATE_QUANTIZATION),
            round(position[1] / REPEATED_STATE_QUANTIZATION),
        )

    return (
        call.scenario_id,
        call.failure_reason,
        call.mapped_robot_grid_cell,
        quantize(call.actual_robot_world_position),
        quantize(call.pedestrian_position),
        quantize(call.pedestrian_velocity),
    )


def _serialized_planning_calls(
    calls: list[SpaceTimePlanningCall],
    method: str,
) -> list[dict[str, object]]:
    seen: set[tuple[object, ...]] = set()
    records = []
    for call in calls:
        record = asdict(call)
        record["method"] = method
        if not call.is_initial_plan and call.failure_reason is not None:
            signature = _failure_state_signature(call)
            record["effectively_same_failure_previously_seen"] = (
                signature in seen
            )
            seen.add(signature)
        else:
            record["effectively_same_failure_previously_seen"] = False
        records.append(record)
    return records


def _first_reactive_block(
    speed_scales: tuple[float, ...],
) -> tuple[int | None, float | None]:
    for index, speed_scale in enumerate(speed_scales):
        if abs(speed_scale) <= STOPPED_SPEED_TOLERANCE:
            return index + 1, index * SIMULATION_STEP
    return None, None


def _result_payload(result: object) -> dict[str, object]:
    return asdict(result)


def _dominant_root_cause(
    trace: object,
    episode_failure_reason: str | None,
) -> tuple[str | None, dict[str, object]]:
    failed_calls = _failed_replans(list(trace.space_time_planning_calls))
    counts = Counter(call.failure_reason for call in failed_calls)
    repeated_groups = [
        group
        for group in group_repeated_failed_planning_calls(failed_calls)
        if group.call_count > 1
    ]
    category = next(
        (
            failure_category
            for failure_category, _ in sorted(
                counts.items(),
                key=lambda item: (
                    -item[1],
                    _FAILURE_CATEGORIES.index(item[0]),
                ),
            )
        ),
        None,
    )
    evidence = {
        "failed_replan_categories": {
            reason: counts[reason] for reason in _FAILURE_CATEGORIES
        },
        "mapped_start_artifacts": sum(
            call.mapped_start_artifact for call in failed_calls
        ),
        "mapped_center_in_predicted_collision_failures": sum(
            call.mapped_center_to_pedestrian_distance
            <= HUMAN_COLLISION_DISTANCE
            for call in failed_calls
        ),
        "actual_pose_in_predicted_collision_failures": sum(
            call.actual_robot_pedestrian_distance
            <= HUMAN_COLLISION_DISTANCE
            for call in failed_calls
        ),
        "actual_pose_with_safe_first_action_failures": sum(
            bool(call.actual_pose_safe_first_actions)
            for call in failed_calls
        ),
        "pedestrian_at_target_failures": sum(
            call.pedestrian_at_target for call in failed_calls
        ),
        "repeated_failure_groups": len(repeated_groups),
        "calls_in_repeated_failure_groups": sum(
            group.call_count for group in repeated_groups
        ),
        "episode_outcome_classification": episode_failure_reason,
    }
    return category or episode_failure_reason, evidence


def _episode_method_payload(
    method: str,
    result: object,
    trace: object,
    diagnostic: object | None,
) -> dict[str, object]:
    initial_call = trace.space_time_planning_calls[0]
    first_block_step, first_block_time = _first_reactive_block(
        trace.speed_scales
    )
    dominant_cause, root_cause_evidence = _dominant_root_cause(
        trace,
        None if diagnostic is None else diagnostic.likely_failure_reason,
    )
    exact_zero_steps = sum(scale == 0.0 for scale in trace.speed_scales)
    near_zero_steps = sum(
        abs(scale) <= STOPPED_SPEED_TOLERANCE
        for scale in trace.speed_scales
    )
    positive_near_zero_steps = near_zero_steps - exact_zero_steps
    if diagnostic is not None and method == "social_spacetime_replan":
        if trace.replan_count == 0 and positive_near_zero_steps:
            dominant_cause = "near_zero_speed_not_seen_by_exact_replan_trigger"
        elif trace.replan_count and trace.failed_replans == 0:
            dominant_cause = (
                "reactive_blocking_despite_successful_replanning"
            )
    root_cause_evidence.update(
        {
            "exact_zero_speed_steps": exact_zero_steps,
            "near_zero_speed_steps": near_zero_steps,
            "positive_near_zero_speed_steps": positive_near_zero_steps,
        }
    )
    return {
        "result": _result_payload(result),
        "initial_plan_success": initial_call.goal_reached,
        "initial_path": [
            [state[0], state[1]]
            for state in initial_call.returned_timed_states
        ],
        "initial_timed_states": initial_call.returned_timed_states,
        "initial_actions": initial_call.returned_actions,
        "initial_planned_wait_count": initial_call.planned_wait_count,
        "first_reactive_block_step": first_block_step,
        "first_reactive_block_time": first_block_time,
        "replan_attempts": trace.replan_count,
        "successful_replans": trace.successful_replans,
        "failed_replans": trace.failed_replans,
        "final_robot_position": trace.final_robot_position,
        "final_pedestrian_position": trace.final_pedestrian_position,
        "final_distance_to_goal": (
            None if diagnostic is None else diagnostic.final_distance_to_goal
        ),
        "minimum_human_distance": result.minimum_human_distance,
        "human_collision": result.human_collision,
        "obstacle_collision": result.obstacle_collision,
        "dominant_root_cause": dominant_cause,
        "root_cause_evidence": root_cause_evidence,
    }


def _aggregate_failed_replans(
    calls: list[SpaceTimePlanningCall],
) -> dict[str, object]:
    failed_calls = _failed_replans(calls)
    category_counts = Counter(call.failure_reason for call in failed_calls)
    failures_by_episode = Counter(
        call.scenario_id for call in failed_calls
    )
    groups = group_repeated_failed_planning_calls(failed_calls)
    repeated_groups = [group for group in groups if group.call_count > 1]
    calls_in_repeated_groups = sum(
        group.call_count for group in repeated_groups
    )
    repeated_retry_count = sum(
        group.call_count - 1 for group in repeated_groups
    )
    failure_count = len(failed_calls)

    return {
        "total_failed_replans": failure_count,
        "counts_by_failure_category": {
            reason: category_counts[reason]
            for reason in _FAILURE_CATEGORIES
        },
        "affected_episode_count": len(failures_by_episode),
        "affected_episode_ids": sorted(failures_by_episode),
        "unique_mapped_robot_cells": [
            list(cell)
            for cell in sorted(
                {call.mapped_robot_grid_cell for call in failed_calls}
            )
        ],
        "unique_mapped_robot_cell_count": len(
            {call.mapped_robot_grid_cell for call in failed_calls}
        ),
        "failed_replans_by_episode": dict(
            sorted(failures_by_episode.items())
        ),
        "mean_failed_replans_per_affected_episode": (
            failure_count / len(failures_by_episode)
            if failures_by_episode
            else None
        ),
        "maximum_failed_replans_in_one_episode": max(
            failures_by_episode.values(),
            default=0,
        ),
        "effectively_unique_failure_state_count": len(groups),
        "repeated_failure_group_count": len(repeated_groups),
        "calls_in_repeated_failure_groups": calls_in_repeated_groups,
        "repeated_retry_count_after_first_occurrence": repeated_retry_count,
        "fraction_calls_in_repeated_failure_groups": (
            calls_in_repeated_groups / failure_count
            if failure_count
            else None
        ),
        "largest_repeated_failure_group": max(
            (group.call_count for group in repeated_groups),
            default=0,
        ),
        "repeated_failure_groups": [
            asdict(group) for group in repeated_groups
        ],
        "no_safe_first_action_count": category_counts[
            "no_safe_first_action"
        ],
        "no_safe_first_action_fraction": (
            category_counts["no_safe_first_action"] / failure_count
            if failure_count
            else None
        ),
        "time_horizon_exhaustion_count": category_counts[
            "time_horizon_exhausted"
        ],
        "time_horizon_exhaustion_fraction": (
            category_counts["time_horizon_exhausted"] / failure_count
            if failure_count
            else None
        ),
        "search_exhaustion_count": category_counts["search_exhausted"],
        "search_exhaustion_fraction": (
            category_counts["search_exhausted"] / failure_count
            if failure_count
            else None
        ),
        "calls_without_any_mapped_safe_first_action": sum(
            not call.mapped_pose_safe_first_actions
            for call in failed_calls
        ),
        "mapped_start_artifact_count": sum(
            call.mapped_start_artifact for call in failed_calls
        ),
        "mapped_start_artifacts_by_episode": dict(
            sorted(
                Counter(
                    call.scenario_id
                    for call in failed_calls
                    if call.mapped_start_artifact
                ).items()
            )
        ),
        "mapped_center_in_predicted_collision_count": sum(
            call.mapped_center_to_pedestrian_distance
            <= HUMAN_COLLISION_DISTANCE
            for call in failed_calls
        ),
        "actual_pose_in_predicted_collision_count": sum(
            call.actual_robot_pedestrian_distance
            <= HUMAN_COLLISION_DISTANCE
            for call in failed_calls
        ),
        "actual_pose_with_safe_first_action_count": sum(
            bool(call.actual_pose_safe_first_actions)
            for call in failed_calls
        ),
        "mean_actual_to_mapped_center_distance": (
            sum(
                call.actual_to_mapped_center_distance
                for call in failed_calls
            )
            / failure_count
            if failure_count
            else None
        ),
        "maximum_actual_to_mapped_center_distance": max(
            (
                call.actual_to_mapped_center_distance
                for call in failed_calls
            ),
            default=0.0,
        ),
        "pedestrian_at_target_count": sum(
            call.pedestrian_at_target for call in failed_calls
        ),
        "pedestrian_at_target_fraction": (
            sum(call.pedestrian_at_target for call in failed_calls)
            / failure_count
            if failure_count
            else None
        ),
    }


def _representative_failures(
    calls: list[SpaceTimePlanningCall],
    method: str,
) -> list[dict[str, object]]:
    representatives = []
    for category in _FAILURE_CATEGORIES:
        match = next(
            (
                call
                for call in calls
                if (
                    not call.is_initial_plan
                    and call.failure_reason == category
                )
            ),
            None,
        )
        if match is not None:
            record = asdict(match)
            record["method"] = method
            representatives.append(record)
    return representatives


def run_diagnosis(args: argparse.Namespace) -> dict[str, object]:
    started_at = perf_counter()
    scenarios = (
        generate_scenarios(args.episodes, args.seed)
        if args.scenario_mode == "controlled"
        else generate_diverse_scenarios(args.episodes, args.seed)
    )
    methods = (
        args.method,
        *(
            method
            for method in _SPACE_TIME_METHODS
            if method != args.method
        ),
    )
    runs: dict[
        str,
        dict[str, tuple[object, object, object | None]],
    ] = {method: {} for method in methods}
    primary_calls: list[SpaceTimePlanningCall] = []

    for scenario in scenarios:
        for method in methods:
            result, trace = run_episode_with_trace(scenario, method)
            diagnostic = diagnose_failure(
                scenario,
                method,
                result,
                trace,
            )
            runs[method][scenario.scenario_id] = (
                result,
                trace,
                diagnostic,
            )
            if method == args.method:
                primary_calls.extend(trace.space_time_planning_calls)

    scenario_by_id = {
        scenario.scenario_id: scenario for scenario in scenarios
    }
    requested_residual_ids = [
        scenario_id
        for scenario_id in _RESIDUAL_EPISODE_IDS
        if scenario_id in scenario_by_id
    ]
    episode_payloads = []
    for scenario_id in requested_residual_ids:
        scenario = scenario_by_id[scenario_id]
        method_payloads = {}
        for method in _SPACE_TIME_METHODS:
            result, trace, diagnostic = runs[method][scenario_id]
            method_payloads[method] = _episode_method_payload(
                method,
                result,
                trace,
                diagnostic,
            )
        episode_payloads.append(
            {
                "scenario_id": scenario_id,
                "scenario": asdict(scenario),
                **method_payloads,
            }
        )

    failure_sets = {}
    for method in _SPACE_TIME_METHODS:
        failure_sets[method] = sorted(
            scenario_id
            for scenario_id, (result, _, _) in runs[method].items()
            if not result.success
        )

    aggregate = _aggregate_failed_replans(primary_calls)
    aggregate["one_shot_vs_replan"] = {
        "social_spacetime_failure_ids": failure_sets[
            "social_spacetime"
        ],
        "social_spacetime_replan_failure_ids": failure_sets[
            "social_spacetime_replan"
        ],
        "same_failure_set": (
            failure_sets["social_spacetime"]
            == failure_sets["social_spacetime_replan"]
        ),
        "residual_episode_count": len(requested_residual_ids),
    }
    aggregate["runtime_seconds"] = perf_counter() - started_at

    move_duration = scenarios[0].grid_scale / ROBOT_SPEED
    return {
        "configuration": {
            "episodes": args.episodes,
            "seed": args.seed,
            "scenario_mode": args.scenario_mode,
            "method": args.method,
            "comparison_method": next(
                method for method in methods if method != args.method
            ),
            "simulation_dt": SIMULATION_STEP,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "max_episode_seconds": MAX_EPISODE_STEPS * SIMULATION_STEP,
            "robot_speed": ROBOT_SPEED,
            "move_duration_seconds": move_duration,
            "social_distance": SOCIAL_DISTANCE,
            "social_weight": SOCIAL_WEIGHT,
            "collision_distance": HUMAN_COLLISION_DISTANCE,
            "failure_categories": list(_FAILURE_CATEGORIES),
            "failure_category_precedence": list(_FAILURE_CATEGORIES),
            "repeated_state_quantization": REPEATED_STATE_QUANTIZATION,
            "mapping_artifact_rule": (
                "failed mapped start has no safe first action while the "
                "actual continuous pose has at least one safe first action"
            ),
            "requested_residual_episode_ids": list(
                _RESIDUAL_EPISODE_IDS
            ),
        },
        "aggregate": aggregate,
        "episodes": episode_payloads,
        "planning_calls": _serialized_planning_calls(
            primary_calls,
            args.method,
        ),
        "representative_failures": _representative_failures(
            primary_calls,
            args.method,
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = run_diagnosis(args)
    output_path = (
        PROJECT_ROOT
        / "outputs"
        / f"spacetime_failure_diagnosis_seed{args.seed}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    aggregate = payload["aggregate"]
    print("Space-Time Residual Failure Diagnosis")
    print("------------------------------------")
    print(f"Failed replans: {aggregate['total_failed_replans']}")
    print(
        "Failure categories: "
        f"{aggregate['counts_by_failure_category']}"
    )
    print(
        "Mapped-start artifacts: "
        f"{aggregate['mapped_start_artifact_count']}"
    )
    print(
        "Pedestrian-at-target failures: "
        f"{aggregate['pedestrian_at_target_count']}"
    )
    print(
        "Repeated retries after first occurrence: "
        f"{aggregate['repeated_retry_count_after_first_occurrence']}"
    )
    print(f"Saved: {output_path}")
    print(f"Runtime: {aggregate['runtime_seconds']:.3f} s")


if __name__ == "__main__":
    main()
