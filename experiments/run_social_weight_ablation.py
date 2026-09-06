"""Fixed social-weight ablation for Robust Space-Time Social (Phase 7C).

This is a diagnostic ablation, not parameter tuning: the frozen benchmark
configuration is untouched.  Only the robust runner's social weight is
temporarily overridden per run via module-attribute substitution, and every
other parameter stays at its frozen value.  No "new best weight" is chosen
and nothing is changed in the main method afterwards.
"""

from __future__ import annotations

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
    MAX_EPISODE_STEPS,
    aggregate_results,
    generate_diverse_scenarios,
    run_episode_with_trace,
)
from socialnav.benchmark import robust_space_time_runner  # noqa: E402
from socialnav.env.world import (  # noqa: E402
    HUMAN_COLLISION_DISTANCE,
    ROBOT_SPEED,
    SIMULATION_STEP,
)

WEIGHTS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
DENSITIES = (1, 3, 5)
EPISODES = 100
SEED = 42
METHOD = "social_spacetime_robust"


def main() -> None:
    started_at = perf_counter()
    frozen_weight = robust_space_time_runner.SOCIAL_WEIGHT
    print(f"Frozen SOCIAL_WEIGHT in robust runner: {frozen_weight}")

    per_weight: dict[str, object] = {}
    try:
        for weight in WEIGHTS:
            weight_key = f"{weight:g}"
            per_density: dict[str, object] = {}
            robust_space_time_runner.SOCIAL_WEIGHT = weight
            for density in DENSITIES:
                density_started = perf_counter()
                scenarios = generate_diverse_scenarios(
                    EPISODES,
                    SEED,
                    pedestrian_count=density,
                )
                results = []
                traces = []
                for scenario in scenarios:
                    result, trace = run_episode_with_trace(
                        scenario,
                        METHOD,
                    )
                    results.append(result)
                    traces.append(trace)
                elapsed = perf_counter() - density_started
                summary = aggregate_results(results)
                per_density[str(density)] = {
                    "pedestrian_count": density,
                    "summary": asdict(summary),
                    "timeout_count": sum(
                        trace.timed_out for trace in traces
                    ),
                    "runtime_seconds": round(elapsed, 3),
                }
                print(
                    f"weight={weight:>4g} N={density}: "
                    f"success={summary.success_rate:.3f} "
                    f"collision={summary.collision_rate:.3f} "
                    f"SPL={summary.mean_spl:.3f} "
                    f"timeouts={per_density[str(density)]['timeout_count']} "
                    f"({elapsed:.1f}s)"
                )
            per_weight[weight_key] = {
                "social_weight": weight,
                "per_density": per_density,
            }
    finally:
        robust_space_time_runner.SOCIAL_WEIGHT = frozen_weight

    payload = {
        "configuration": {
            "phase": "7C social-weight ablation",
            "method": METHOD,
            "seed": SEED,
            "episodes_per_density": EPISODES,
            "scenario_mode": "diverse",
            "social_weights": list(WEIGHTS),
            "densities": list(DENSITIES),
            "frozen_social_weight_in_main_benchmark": frozen_weight,
            "ablation_note": (
                "diagnostic only; the main benchmark configuration and "
                "the frozen robust planner are unchanged"
            ),
            "simulation_dt": SIMULATION_STEP,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "max_episode_seconds": MAX_EPISODE_STEPS * SIMULATION_STEP,
            "replan_stop_steps": REPLAN_STOP_STEPS,
            "replan_stop_seconds": REPLAN_STOP_SECONDS,
            "human_collision_distance": HUMAN_COLLISION_DISTANCE,
            "robot_speed": ROBOT_SPEED,
        },
        "per_weight": per_weight,
    }
    output_path = PROJECT_ROOT / "outputs" / "social_weight_ablation_seed42.json"
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