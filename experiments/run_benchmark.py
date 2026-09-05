"""Run the reproducible five-method SocialNav benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
)
from socialnav.env.world import SIMULATION_STEP  # noqa: E402
from socialnav.planners import ESCAPE_SPEED_SCALE  # noqa: E402

_METHOD_LABELS = {
    "astar": "A*",
    "dynamic": "Dynamic",
    "social": "Social",
    "social_replan": "Social Replan",
    "social_replan_escape": "Social Replan + Escape",
}
_SCENARIO_MODES = ("controlled", "diverse")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print_table(summaries: dict[str, MethodSummary]) -> None:
    print(
        "Method                  Success  Collision  SPL    PathLen  Time   "
        "MinHumanDist  SocialViolation"
    )
    print(
        "----------------------  -------  ---------  -----  -------  -----  "
        "------------  ---------------"
    )
    for method in SUPPORTED_METHODS:
        summary = summaries[method]
        print(
            f"{_METHOD_LABELS[method]:<22}  "
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

    started_at = perf_counter()
    if args.scenario_mode == "controlled":
        scenarios = generate_scenarios(args.episodes, args.seed)
    else:
        scenarios = generate_diverse_scenarios(args.episodes, args.seed)
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
        "configuration": {
            "simulation_dt": SIMULATION_STEP,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "max_episode_seconds": MAX_EPISODE_STEPS * SIMULATION_STEP,
            "replan_stop_seconds": REPLAN_STOP_SECONDS,
            "replan_stop_steps": REPLAN_STOP_STEPS,
            "escape_speed_scale": ESCAPE_SPEED_SCALE,
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
            },
        },
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "methods": {
            method: {
                "summary": asdict(summaries[method]),
                "replanning": _summarize_replans(
                    traces_by_method[method]
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
