"""Run the reproducible three-method SocialNav benchmark."""

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
    MAX_EPISODE_STEPS,
    SUPPORTED_METHODS,
    aggregate_results,
    generate_scenarios,
    run_episode,
)
from socialnav.env.world import SIMULATION_STEP  # noqa: E402

_METHOD_LABELS = {
    "astar": "A*",
    "dynamic": "Dynamic",
    "social": "Social",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print_table(summaries: dict[str, object]) -> None:
    print(
        "Method    Success  Collision  SPL    PathLen  Time   "
        "MinHumanDist  SocialViolation"
    )
    print(
        "--------  -------  ---------  -----  -------  -----  "
        "------------  ---------------"
    )
    for method in SUPPORTED_METHODS:
        summary = summaries[method]
        print(
            f"{_METHOD_LABELS[method]:<8}  "
            f"{summary.success_rate:>7.3f}  "
            f"{summary.collision_rate:>9.3f}  "
            f"{summary.mean_spl:>5.3f}  "
            f"{summary.mean_path_length:>7.3f}  "
            f"{_format_optional(summary.mean_time_to_goal):>5}  "
            f"{_format_optional(summary.mean_minimum_human_distance):>12}  "
            f"{summary.mean_social_violation_rate:>15.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic SocialNav headless benchmark."
    )
    parser.add_argument("--episodes", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started_at = perf_counter()
    scenarios = generate_scenarios(args.episodes, args.seed)
    results_by_method = {method: [] for method in SUPPORTED_METHODS}

    for scenario in scenarios:
        for method in SUPPORTED_METHODS:
            results_by_method[method].append(
                run_episode(scenario, method)
            )

    summaries = {
        method: aggregate_results(results)
        for method, results in results_by_method.items()
    }
    _print_table(summaries)

    output_path = (
        PROJECT_ROOT / "outputs" / f"benchmark_seed{args.seed}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "episodes": args.episodes,
        "configuration": {
            "simulation_dt": SIMULATION_STEP,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "max_episode_seconds": MAX_EPISODE_STEPS * SIMULATION_STEP,
            "methods": {
                "astar": "ordinary A* without reactive avoidance",
                "dynamic": "ordinary A* with reactive avoidance",
                "social": "social A* with reactive avoidance",
            },
        },
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "methods": {
            method: {
                "summary": asdict(summaries[method]),
                "episodes": [
                    {
                        "scenario_id": scenario.scenario_id,
                        "result": asdict(result),
                    }
                    for scenario, result in zip(
                        scenarios,
                        results_by_method[method],
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
