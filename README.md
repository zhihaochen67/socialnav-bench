# SocialNav-Bench

A reproducible benchmark for socially-aware robot navigation in dynamic multi-pedestrian environments.

> **Status:** Active development

SocialNav-Bench studies how classical, social-aware, predictive, and space-time planners behave as pedestrian density increases. The project combines navigation algorithms, continuous execution, safety-aware local control, reproducible benchmarking, and systematic failure analysis.

## Highlights

- Reproducible dynamic pedestrian navigation benchmark
- Classical A* and reactive dynamic-avoidance baselines
- Social-aware and predictive social planning
- Space-Time Social A* with explicit MOVE / WAIT reasoning
- Robust continuous execution with replanning and collision egress
- Predictive local safety shield for multi-human navigation
- Arbitrary multi-pedestrian scenarios
- Density benchmarks at 1 / 3 / 5 / 10 pedestrians
- Failure taxonomy, diagnostics, and social-weight ablations
- 477 automated tests

## Benchmark Snapshot

100 diverse episodes per density, seed 42.

| Pedestrians | Method | Success | Collision | SPL | Min Human Distance | Social Violation |
|---:|---|---:|---:|---:|---:|---:|
| 3 | Robust Space-Time | 81% | 18% | 0.671 | 0.502 | 0.351 |
| 3 | **Shielded Space-Time** | **84%** | **0%** | **0.678** | **0.644** | **0.102** |
| 5 | Robust Space-Time | 38% | 23% | 0.303 | 0.486 | 0.452 |
| 5 | **Shielded Space-Time** | **47%** | **1%** | **0.344** | **0.648** | **0.104** |
| 10 | Robust Space-Time | 11% | 43% | 0.089 | 0.397 | 0.360 |
| 10 | **Shielded Space-Time** | **18%** | **7%** | **0.123** | **0.549** | **0.117** |

The predictive local safety shield substantially reduces collisions in dense multi-pedestrian scenarios while preserving the underlying robust space-time planner.

## Navigation Methods

The benchmark currently includes:

- `astar`
- `dynamic`
- `social`
- `social_replan`
- `social_replan_escape`
- `social_replan_recovery`
- `social_predictive`
- `social_predictive_replan`
- `social_spacetime`
- `social_spacetime_replan`
- `social_spacetime_robust`
- `social_spacetime_shielded`

The latest method combines robust space-time planning with predictive local MOVE / WAIT safety checks, deterministic physical overrides, and immediate post-override replanning.

## Evaluation Metrics

SocialNav-Bench evaluates navigation using:

- Success rate
- Collision rate
- Path length
- Time to goal
- SPL
- Minimum human distance
- Social violation rate

## Project Structure

```text
socialnav-bench/
├── socialnav/       # planners, simulation, benchmark and evaluation code
├── experiments/     # benchmark, ablation and failure-analysis scripts
├── tests/           # regression and behavior tests
├── outputs/         # generated experiment artifacts (gitignored)
└── README.md
```

## Run the Tests

From the project root:

```bash
python -m pytest -q
```

Current verified state:

```text
477 passed
```

## Reproducible Experiments

Main experiment entry points include:

```text
experiments/run_benchmark.py
experiments/run_density_benchmark.py
experiments/run_social_weight_ablation.py
experiments/analyze_failures.py
experiments/analyze_multi_human_failures.py
experiments/diagnose_spacetime_failures.py
```

The benchmark uses deterministic seeds and fixed scenario-generation semantics so planner comparisons can be reproduced consistently.

## Current Research Direction

The latest failure analysis shows that predictive local shielding removes most stationary pedestrian-into-robot collisions. The remaining dense-crowd failures are dominated by post-override replanning and finite-horizon planning limitations.

This project is still under active development. Additional visualization, documentation, experiment summaries, and final reproducibility instructions will be added before the first stable release.

## Tech Stack

Python · PyBullet · NumPy · Matplotlib · Pandas · pytest

## Author

**Zhihao Chen**

Computer Science, Nanjing University of Posts and Telecommunications
