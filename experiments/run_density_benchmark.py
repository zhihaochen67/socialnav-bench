"""Reproducible multi-pedestrian density benchmark (1/3/5/10 pedestrians)."""

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
    MAX_EPISODE_STEPS,
    REPLAN_STOP_SECONDS,
    REPLAN_STOP_STEPS,
    EpisodeTrace,
    MethodSummary,
    SUPPORTED_METHODS,
    aggregate_results,
    generate_diverse_scenarios,
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

DEFAULT_PEDESTRIAN_COUNTS = (1, 3, 5, 10)

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

_METHOD_NAMES = list(SUPPORTED_METHODS)


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


def _pedestrian_counts(value: str) -> list[int]:
    counts = []
    for token in value.split(","):
        parsed = _non_negative_int(token.strip())
        if parsed in counts:
            raise argparse.ArgumentTypeError(
                f"duplicate pedestrian count {parsed}"
            )
        counts.append(parsed)
    if not counts:
        raise argparse.ArgumentTypeError("at least one count is required")
    return counts


def _method_names(value: str) -> list[str]:
    methods = []
    for token in value.split(","):
        name = token.strip()
        if name not in SUPPORTED_METHODS:
            raise argparse.ArgumentTypeError(
                f"unsupported method {name!r}; must be one of "
                f"{', '.join(SUPPORTED_METHODS)}"
            )
        if name not in methods:
            methods.append(name)
    if not methods:
        raise argparse.ArgumentTypeError("at least one method is required")
    return methods


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _search_statistics(traces: list[EpisodeTrace]) -> dict[str, int | None]:
    """Aggregate space-time search expansion evidence where available."""
    expanded = [
        call.expanded_states
        for trace in traces
        for call in trace.space_time_planning_calls
    ]
    generated = [
        call.generated_states
        for trace in traces
        for call in trace.space_time_planning_calls
    ]
    if not expanded:
        return {
            "mean_expanded_states": None,
            "max_expanded_states": None,
            "mean_generated_states": None,
            "max_generated_states": None,
        }
    return {
        "mean_expanded_states": sum(expanded) / len(expanded),
        "max_expanded_states": max(expanded),
        "mean_generated_states": sum(generated) / len(generated),
        "max_generated_states": max(generated),
    }


def _print_density_table(
    pedestrian_count: int,
    summaries: dict[str, MethodSummary],
    timeout_counts: dict[str, int],
    methods: list[str],
) -> None:
    print()
    print(f"Pedestrians={pedestrian_count}")
    print(
        "Pedestrians | Method                     | Success | Collision | "
        "SPL    | MinDist | SocialViolation | Timeouts"
    )
    print(
        "----------- | -------------------------- | ------- | --------- | "
        "------ | ------- | --------------- | --------"
    )
    for method in methods:
        summary = summaries[method]
        print(
            f"{pedestrian_count:<11} | {_METHOD_LABELS[method]:<26} | "
            f"{summary.success_rate:>7.3f} | "
            f"{summary.collision_rate:>9.3f} | "
            f"{summary.mean_spl:>6.3f} | "
            f"{_format_optional(summary.mean_minimum_human_distance):>7} | "
            f"{summary.mean_social_violation_rate:>15.3f} | "
            f"{timeout_counts[method]:>8}"
        )


def _print_robust_trend(
    robust_rows: dict[int, object],
) -> None:
    print()
    print("Robust Space-Time Social trend by pedestrian count:")
    print(
        "Pedestrians | Success | Collision | SPL    | MinDist | "
        "SocialViolation | Timeouts"
    )
    print(
        "----------- | ------- | --------- | ------ | ------- | "
        "--------------- | --------"
    )
    for pedestrian_count in sorted(robust_rows):
        row = robust_rows[pedestrian_count]
        assert isinstance(row, dict)
        print(
            f"{pedestrian_count:<11} | {row['success_rate']:>7.3f} | "
            f"{row['collision_rate']:>9.3f} | {row['mean_spl']:>6.3f} | "
            f"{_format_optional(row['mean_minimum_human_distance']):>7} | "
            f"{row['mean_social_violation_rate']:>15.3f} | "
            f"{row['timeout_count']:>8}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic multi-pedestrian density benchmark "
            "with reproducible JSON output."
        )
    )
    parser.add_argument("--episodes", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pedestrians",
        type=_pedestrian_counts,
        default=list(DEFAULT_PEDESTRIAN_COUNTS),
        help=(
            "comma-separated pedestrian counts "
            f"(default: {','.join(map(str, DEFAULT_PEDESTRIAN_COUNTS))})"
        ),
    )
    parser.add_argument(
        "--methods",
        type=_method_names,
        default=_METHOD_NAMES,
        help=(
            "comma-separated methods "
            f"(default: all {len(_METHOD_NAMES)} supported methods)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output JSON path (default: outputs/density_benchmark_seed<seed>.json)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pedestrian_counts = args.pedestrians
    methods = args.methods
    seed = args.seed
    episodes = args.episodes

    started_at = perf_counter()
    spacetime_move_duration = 0.75 / ROBOT_SPEED
    configuration = {
        "seed": seed,
        "episodes_per_density": episodes,
        "scenario_mode": "diverse",
        "pedestrian_counts": pedestrian_counts,
        "methods": methods,
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
        "spacetime_move_duration_seconds": spacetime_move_duration,
        "spacetime_action_steps": duration_to_simulation_steps(
            spacetime_move_duration,
            SIMULATION_STEP,
        ),
        "spacetime_max_time_index": floor(
            MAX_EPISODE_STEPS
            * SIMULATION_STEP
            / spacetime_move_duration
        ),
        "robust_stall_window_steps": REPLAN_STOP_STEPS,
        "robust_progress_threshold_meters": ROBOT_SPEED * SIMULATION_STEP,
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
                "predictive social A* with reactive avoidance and online "
                "replanning using current pedestrian velocities"
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
    }

    per_density: dict[str, object] = {}
    robust_trend: dict[int, object] = {}

    for pedestrian_count in pedestrian_counts:
        density_started = perf_counter()
        print(
            f"\nGenerating {episodes} diverse scenarios with "
            f"{pedestrian_count} pedestrian(s), seed={seed}"
        )
        scenarios = generate_diverse_scenarios(
            episodes,
            seed,
            pedestrian_count=pedestrian_count,
        )
        results_by_method: dict[str, list[EpisodeResult]] = {}
        traces_by_method: dict[str, list[EpisodeTrace]] = {}
        runtimes: dict[str, float] = {}
        summaries: dict[str, MethodSummary] = {}
        timeout_counts: dict[str, int] = {}

        for method in methods:
            method_started = perf_counter()
            results: list[EpisodeResult] = []
            traces: list[EpisodeTrace] = []
            for scenario in scenarios:
                result, trace = run_episode_with_trace(scenario, method)
                results.append(result)
                traces.append(trace)
            elapsed = perf_counter() - method_started
            results_by_method[method] = results
            traces_by_method[method] = traces
            runtimes[method] = elapsed
            summaries[method] = aggregate_results(results)
            timeout_counts[method] = sum(
                trace.timed_out for trace in traces
            )
            print(
                f"  {_METHOD_LABELS[method]:<26} "
                f"success={summaries[method].success_rate:.3f} "
                f"collision={summaries[method].collision_rate:.3f} "
                f"runtime={elapsed:.1f}s"
            )

        per_method: dict[str, object] = {}
        for method in methods:
            summary = summaries[method]
            per_method[method] = {
                "summary": asdict(summary),
                "timeout_count": timeout_counts[method],
                "runtime_seconds": round(runtimes[method], 3),
                "search_statistics": _search_statistics(
                    traces_by_method[method]
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
            }

        density_elapsed = perf_counter() - density_started
        per_density[str(pedestrian_count)] = {
            "pedestrian_count": pedestrian_count,
            "runtime_seconds": round(density_elapsed, 3),
            "episodes": episodes,
            "per_method": per_method,
        }
        _print_density_table(
            pedestrian_count,
            summaries,
            timeout_counts,
            methods,
        )
        print(
            f"  density runtime: {density_elapsed:.1f}s "
            f"(total episodes: {episodes * len(methods)})"
        )

        if "social_spacetime_robust" in methods:
            robust_summary = summaries["social_spacetime_robust"]
            robust_diagnostics = per_method["social_spacetime_robust"][
                "robust"
            ]
            assert isinstance(robust_diagnostics, dict)
            robust_trend[pedestrian_count] = {
                "success_rate": robust_summary.success_rate,
                "collision_rate": robust_summary.collision_rate,
                "mean_spl": robust_summary.mean_spl,
                "mean_minimum_human_distance": (
                    robust_summary.mean_minimum_human_distance
                ),
                "mean_social_violation_rate": (
                    robust_summary.mean_social_violation_rate
                ),
                "timeout_count": timeout_counts[
                    "social_spacetime_robust"
                ],
                "mean_replan_count": robust_diagnostics[
                    "robust_replan_count"
                ],
                "continuous_bridge_attempts": robust_diagnostics[
                    "continuous_bridge_attempts"
                ],
                "continuous_bridge_successes": robust_diagnostics[
                    "continuous_bridge_successes"
                ],
                "collision_egress_attempts": robust_diagnostics[
                    "collision_egress_attempts"
                ],
                "progress_stall_events": robust_diagnostics[
                    "progress_stall_events"
                ],
                "exact_zero_stall_events": robust_diagnostics[
                    "exact_zero_stall_events"
                ],
                "planning_failure_reasons": robust_diagnostics[
                    "planning_failure_reasons"
                ],
                "remaining_failures": robust_diagnostics[
                    "remaining_failures"
                ],
            }

    output_path = (
        Path(args.output)
        if args.output is not None
        else PROJECT_ROOT
        / "outputs"
        / f"density_benchmark_seed{seed}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "configuration": configuration,
        "pedestrian_counts": pedestrian_counts,
        "per_density": per_density,
        "robust_trend": robust_trend,
    }
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    if robust_trend:
        _print_robust_trend(robust_trend)  # type: ignore[arg-type]

    elapsed = perf_counter() - started_at
    print(f"\nSaved: {output_path}")
    print(f"Total runtime: {elapsed:.1f} s")


if __name__ == "__main__":
    main()