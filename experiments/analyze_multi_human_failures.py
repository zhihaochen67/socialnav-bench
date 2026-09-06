"""Deterministic Phase 7C multi-human failure analysis driver.

Diagnosis only: this script runs the frozen robust space-time method over
the frozen diverse scenarios and classifies/aggregates failures from actual
trace evidence.  It never modifies planner behavior, scenario generation,
or the episode timeout.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
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
    Scenario,
    generate_diverse_scenarios,
    run_episode_with_trace,
)
from socialnav.benchmark.failure_analysis import (  # noqa: E402
    FAILURE_CATEGORIES,
    PLANNING_EXECUTION_CLASSES,
    aggregate_features,
    analyze_first_collision,
    analyze_local_feasibility,
    classify_planning_vs_execution,
    classify_robust_failure,
    compute_final_state_geometry,
    compute_terminal_blockage_evidence,
    scenario_difficulty_features,
    select_representative_cases,
    summarize_search_statistics,
    summarize_trace_counters,
)
from socialnav.benchmark.robust_failure_probe import (  # noqa: E402
    run_robust_episode_with_evidence,
)
from socialnav.env.world import (  # noqa: E402
    HUMAN_COLLISION_DISTANCE,
    ROBOT_SPEED,
    SIMULATION_STEP,
    SLOW_DISTANCE,
    STOP_DISTANCE,
)
from socialnav.evaluation import EpisodeResult  # noqa: E402

DENSITIES = (3, 5, 10)
EPISODES_PER_DENSITY = 100
SEED = 42
METHOD = "social_spacetime_robust"

_FEATURE_KEYS = (
    "pedestrian_count",
    "astar_path_length_moves",
    "astar_path_length_world",
    "trajectories_intersecting_or_approaching_route",
    "minimum_initial_human_route_distance",
    "route_crossing_pedestrians",
    "pedestrian_targets_near_route",
    "approximate_max_simultaneous_nearby_humans",
)

REPRESENTATIVE_REQUIREMENTS: dict[int, list[dict[str, str]]] = {
    3: [
        {"kind": "collision", "label": "collision"},
        {"kind": "timeout", "label": "timeout"},
    ],
    5: [
        {"kind": "collision", "label": "collision"},
        {"kind": "corridor_blockage", "label": "multi_human_blockage"},
        {"kind": "planning_failure", "label": "planning_replan_failure"},
    ],
    10: [
        {"kind": "collision", "label": "collision"},
        {"kind": "local_trap", "label": "local_trap"},
        {"kind": "terminal_blockage", "label": "terminal_blockage"},
        {"kind": "different_dominant", "label": "different_dominant_failure"},
    ],
}


def _configuration() -> dict[str, object]:
    return {
        "phase": "7C",
        "method": METHOD,
        "seed": SEED,
        "densities": list(DENSITIES),
        "episodes_per_density": EPISODES_PER_DENSITY,
        "scenario_mode": "diverse",
        "simulation_dt": SIMULATION_STEP,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "max_episode_seconds": MAX_EPISODE_STEPS * SIMULATION_STEP,
        "replan_stop_steps": REPLAN_STOP_STEPS,
        "replan_stop_seconds": REPLAN_STOP_SECONDS,
        "human_collision_distance": HUMAN_COLLISION_DISTANCE,
        "stop_distance": STOP_DISTANCE,
        "slow_distance": SLOW_DISTANCE,
        "social_distance": 0.70,
        "robot_speed": ROBOT_SPEED,
        "failure_category_priority": list(FAILURE_CATEGORIES),
        "planning_execution_classes": list(PLANNING_EXECUTION_CLASSES),
        "analysis_rules": (
            "exclusive first-match priority taxonomy over EpisodeResult/"
            "EpisodeTrace evidence; terminal-blockage thresholds are the "
            "existing STOP_DISTANCE/SLOW_DISTANCE constants; collision "
            "attribution uses probe-recorded trajectories"
        ),
    }


def _trace_counters_record(trace: EpisodeTrace) -> dict[str, object]:
    return {
        "continuous_bridge_attempts": trace.continuous_bridge_attempts,
        "continuous_bridge_successes": trace.continuous_bridge_successes,
        "continuous_bridge_failures": trace.continuous_bridge_failures,
        "robust_replan_count": trace.robust_replan_count,
        "robust_replan_successes": trace.robust_replan_successes,
        "robust_replan_failures": trace.robust_replan_failures,
        "exact_zero_stall_events": trace.exact_zero_stall_events,
        "progress_stall_events": trace.progress_stall_events,
        "suppressed_duplicate_replans": (
            trace.suppressed_duplicate_replans
        ),
        "collision_egress_attempts": trace.collision_egress_attempts,
        "collision_egress_successes": trace.collision_egress_successes,
        "collision_egress_failures": trace.collision_egress_failures,
        "robust_planning_failure_reasons": list(
            trace.robust_planning_failure_reasons
        ),
        "spacetime_plan_count": trace.spacetime_plan_count,
        "spacetime_planning_failures": trace.spacetime_planning_failures,
    }


def _analyze_one_density(
    density: int,
) -> dict[str, object]:
    print(f"\n=== Density N={density}: running 100 robust episodes ===")
    scenarios = generate_diverse_scenarios(
        EPISODES_PER_DENSITY,
        SEED,
        pedestrian_count=density,
    )
    results: list[EpisodeResult] = []
    traces: list[EpisodeTrace] = []
    runtimes: list[float] = []
    for scenario in scenarios:
        started = perf_counter()
        result, trace = run_episode_with_trace(scenario, METHOD)
        runtimes.append(perf_counter() - started)
        results.append(result)
        traces.append(trace)

    episode_rows: list[dict[str, object]] = []
    collision_rows: list[dict[str, object]] = []
    failure_category_counts: Counter[str] = Counter()
    planning_execution_counts: Counter[str] = Counter()
    local_trap_count = 0
    local_multi_human_trap_count = 0
    terminal_blocker_failures = 0
    joint_corridor_failures = 0
    failure_count = 0

    for scenario, result, trace in zip(scenarios, results, traces):
        features = scenario_difficulty_features(scenario)
        row: dict[str, object] = {
            "density": density,
            "scenario_id": scenario.scenario_id,
            "success": result.success,
            "timed_out": trace.timed_out,
            "runtime_seconds": None,
            "failure_category": None,
            "planning_execution_class": None,
            "locally_trapped": None,
            "terminal_blockage": None,
            "final_state_geometry": None,
            "collision": None,
            "trace_counters": _trace_counters_record(trace),
            "scenario_features": features,
        }
        if result.success:
            episode_rows.append(row)
            continue

        failure_count += 1
        category = classify_robust_failure(scenario, result, trace)
        planning_class = classify_planning_vs_execution(result, trace)
        failure_category_counts[category] += 1
        planning_execution_counts[planning_class] += 1
        blockage = compute_terminal_blockage_evidence(scenario, trace)
        geometry = compute_final_state_geometry(scenario, result, trace)
        feasibility = analyze_local_feasibility(scenario, trace)
        trapped = bool(feasibility["locally_trapped"])
        multi_human_trap = trapped and bool(feasibility["multi_human_nearby"])
        local_trap_count += int(trapped)
        local_multi_human_trap_count += int(multi_human_trap)
        if blockage.terminal_route_blocker_indices:
            terminal_blocker_failures += 1
        if blockage.joint_corridor_blockage:
            joint_corridor_failures += 1

        row["failure_category"] = category
        row["planning_execution_class"] = planning_class
        row["locally_trapped"] = trapped
        row["terminal_blockage"] = {
            "terminal_indices": list(blockage.terminal_indices),
            "terminal_route_blocker_indices": list(
                blockage.terminal_route_blocker_indices
            ),
            "joint_corridor_blockage": blockage.joint_corridor_blockage,
            "robot_near_blocker": blockage.robot_near_blocker,
            "target_on_or_near_route_count": (
                blockage.target_on_or_near_route_count
            ),
        }
        row["final_state_geometry"] = geometry
        row["local_feasibility_flags"] = {
            "locally_trapped": trapped,
            "no_safe_bridge": feasibility["no_safe_bridge"],
            "no_safe_action": feasibility["no_safe_action"],
            "multi_human_nearby": feasibility["multi_human_nearby"],
            "critical_human_indices": feasibility[
                "critical_human_indices"
            ],
        }
        episode_rows.append(row)

    # Collision attribution via the evidence probe (re-runs each colliding
    # episode deterministically through the official runner).
    collision_episodes = [
        (scenario, result, trace)
        for scenario, result, trace in zip(scenarios, results, traces)
        if result.human_collision
    ]
    for scenario, result, trace in collision_episodes:
        probe_result, probe_trace, evidence = (
            run_robust_episode_with_evidence(scenario)
        )
        if probe_result != result or probe_trace != trace:
            raise RuntimeError(
                f"probe replay diverged from official run for "
                f"{scenario.scenario_id}"
            )
        attribution = analyze_first_collision(
            scenario,
            result,
            trace,
            evidence,
        )
        if attribution is None:
            raise RuntimeError(
                f"collision episode {scenario.scenario_id} yielded no "
                "attribution"
            )
        collision_rows.append(
            {
                "density": density,
                "scenario_id": scenario.scenario_id,
                "failure_category": classify_robust_failure(
                    scenario,
                    result,
                    trace,
                ),
                **attribution,
            }
        )
        for row in episode_rows:
            if row["scenario_id"] == scenario.scenario_id:
                row["collision"] = attribution
                break

    # Representative cases with full evidence.
    selected = select_representative_cases(
        episode_rows,
        {density: REPRESENTATIVE_REQUIREMENTS[density]},
    )
    representative_detail = {}
    for key, selection in selected.items():
        scenario_id = selection["selected"]
        if scenario_id is None:
            representative_detail[key] = {"selected": None}
            continue
        scenario = next(
            candidate
            for candidate in scenarios
            if candidate.scenario_id == scenario_id
        )
        result = results[scenarios.index(scenario)]
        trace = traces[scenarios.index(scenario)]
        feasibility = analyze_local_feasibility(scenario, trace)
        detail: dict[str, object] = {
            "selected": scenario_id,
            "density": density,
            "kind": selection["kind"],
            "failure_category": classify_robust_failure(
                scenario,
                result,
                trace,
            ),
            "planning_execution_class": classify_planning_vs_execution(
                result,
                trace,
            ),
            "final_state_geometry": compute_final_state_geometry(
                scenario,
                result,
                trace,
            ),
            "local_feasibility": feasibility,
            "terminal_blockage": {
                key_: list(value)
                if isinstance(value, tuple)
                else value
                for key_, value in vars(
                    compute_terminal_blockage_evidence(scenario, trace)
                ).items()
            },
            "trace_counters": _trace_counters_record(trace),
        }
        if result.human_collision:
            _, _, evidence = run_robust_episode_with_evidence(scenario)
            detail["collision"] = analyze_first_collision(
                scenario,
                result,
                trace,
                evidence,
            )
        representative_detail[key] = detail

    feature_rows = [row["scenario_features"] for row in episode_rows]
    failed_feature_rows = [
        row["scenario_features"]
        for row in episode_rows
        if not row["success"]
    ]
    successful_feature_rows = [
        row["scenario_features"]
        for row in episode_rows
        if row["success"]
    ]

    density_report = {
        "pedestrian_count": density,
        "episodes": len(episode_rows),
        "failures": failure_count,
        "successes": len(episode_rows) - failure_count,
        "runtime_seconds": round(sum(runtimes), 3),
        "mean_episode_runtime_seconds": (
            sum(runtimes) / len(runtimes)
        ),
        "failure_counts_by_category": {
            category: failure_category_counts.get(category, 0)
            for category in FAILURE_CATEGORIES
        },
        "planning_vs_execution": {
            label: planning_execution_counts.get(label, 0)
            for label in PLANNING_EXECUTION_CLASSES
        },
        "trace_counters": summarize_trace_counters(results, traces),
        "search_statistics": summarize_search_statistics(results, traces),
        "terminal_blockage_statistics": {
            "failures_with_terminal_route_blocker": (
                terminal_blocker_failures
            ),
            "failures_with_joint_corridor_blockage": (
                joint_corridor_failures
            ),
            "terminal_blocker_failure_fraction": (
                terminal_blocker_failures / failure_count
                if failure_count
                else None
            ),
            "joint_corridor_failure_fraction": (
                joint_corridor_failures / failure_count
                if failure_count
                else None
            ),
        },
        "local_trap_statistics": {
            "locally_trapped_failures": local_trap_count,
            "multi_human_locally_trapped_failures": (
                local_multi_human_trap_count
            ),
            "locally_trapped_failure_fraction": (
                local_trap_count / failure_count
                if failure_count
                else None
            ),
        },
        "scenario_features": {
            "successes": aggregate_features(
                successful_feature_rows,
                _FEATURE_KEYS,
            ),
            "failures": aggregate_features(
                failed_feature_rows,
                _FEATURE_KEYS,
            ),
        },
        "collision_statistics": {
            "collision_episodes": len(collision_rows),
            "successful_collision_episodes": sum(
                row["success"]
                for row in episode_rows
                if row.get("collision") is not None
            ),
            "phase_counts": dict(
                Counter(row["phase"] for row in collision_rows)
            ),
            "robot_stopped_at_collision": sum(
                row["robot_stopped"] for row in collision_rows
            ),
            "pedestrian_moved_into_robot": sum(
                row["pedestrian_moved_into_robot"] for row in collision_rows
            ),
            "mean_first_collision_seconds": (
                sum(
                    row["first_collision_seconds"]
                    for row in collision_rows
                )
                / len(collision_rows)
                if collision_rows
                else None
            ),
        },
        "episodes": episode_rows,
        "collision_episode_details": collision_rows,
        "representative_cases": representative_detail,
    }
    return density_report
def _print_failure_table(
    density_report: dict[str, object],
) -> None:
    density = density_report["pedestrian_count"]
    print(f"\nFailure categories (N={density}):")
    for category in FAILURE_CATEGORIES:
        count = density_report["failure_counts_by_category"][category]
        if count:
            print(f"  {category:<38} {count:>4}")
    print("  planning-vs-execution:")
    for label in PLANNING_EXECUTION_CLASSES:
        count = density_report["planning_vs_execution"][label]
        if count:
            print(f"    {label:<46} {count:>4}")
    print("  representative cases:")
    for key, detail in density_report["representative_cases"].items():
        print(f"    {key}: {detail['selected']}")


def main() -> None:
    started_at = perf_counter()
    per_density: dict[str, object] = {}
    for density in DENSITIES:
        report = _analyze_one_density(density)
        per_density[str(density)] = report
        _print_failure_table(report)

    global_category_counts: Counter[str] = Counter()
    global_planning_counts: Counter[str] = Counter()
    for density in DENSITIES:
        report = per_density[str(density)]
        for category in FAILURE_CATEGORIES:
            global_category_counts[category] += report[
                "failure_counts_by_category"
            ][category]
        for label in PLANNING_EXECUTION_CLASSES:
            global_planning_counts[label] += report[
                "planning_vs_execution"
            ][label]

    payload = {
        "configuration": _configuration(),
        "per_density": per_density,
        "failure_categories": {
            category: global_category_counts.get(category, 0)
            for category in FAILURE_CATEGORIES
        },
        "planning_vs_execution": {
            label: global_planning_counts.get(label, 0)
            for label in PLANNING_EXECUTION_CLASSES
        },
        "episodes": [
            row
            for density in DENSITIES
            for row in per_density[str(density)]["episodes"]
        ],
        "search_statistics": {
            str(density): per_density[str(density)]["search_statistics"]
            for density in DENSITIES
        },
        "collision_statistics": {
            str(density): per_density[str(density)][
                "collision_statistics"
            ]
            for density in DENSITIES
        },
        "terminal_blockage_statistics": {
            str(density): per_density[str(density)][
                "terminal_blockage_statistics"
            ]
            for density in DENSITIES
        },
        "representative_cases": {
            key: detail
            for density in DENSITIES
            for key, detail in per_density[str(density)][
                "representative_cases"
            ].items()
            if key.startswith(f"n{density}_")
        },
    }
    output_path = (
        PROJECT_ROOT / "outputs" / "multi_human_failure_analysis_seed42.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    elapsed = perf_counter() - started_at
    print(f"\nSaved: {output_path}")
    print(f"Runtime: {elapsed:.1f} s")


if __name__ == "__main__":
    main()