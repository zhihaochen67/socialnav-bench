# SocialNav-Bench

> **Status:** v1.0.0

SocialNav-Bench is a reproducible simulation benchmark for socially-aware robot
navigation in dynamic multi-pedestrian environments. It covers classical
planning, predictive and space-time reasoning, robust execution, safety
shielding, evaluation, and systematic failure analysis.

Python 3.10+ · PyBullet · Matplotlib · pytest

## Why it matters

Navigation methods that work in static maps can fail when people move through,
occupy, or block the same space as the robot. SocialNav-Bench makes those
failures measurable across deterministic, seeded scenarios, then preserves the
evidence needed to distinguish planning limits from persistent environmental
blockages.

## Demo

This representative replay uses the same frozen seeded scenario and unchanged
initial conditions for both methods:
`diverse-seed-42-episode-0024-pedestrians-3` (`N=3`). Robust Space-Time Social
collides and then times out; Shielded Space-Time Social avoids the collision and
succeeds. This single episode illustrates the safety shield's behavior—it is
not a summary of every benchmark episode.

https://github.com/user-attachments/assets/5b87a529-a890-437a-a3f0-883b893f7c42

![Trajectory comparison for the same frozen demo episode](docs/assets/trajectory_comparison.png)

## Headline benchmark results

The frozen Phase 7D comparison uses 100 diverse episodes per density with seed
42. The predictive local safety shield reduces collision rate from **18% to 0%**
at `N=3`, **23% to 1%** at `N=5`, and **43% to 7%** at `N=10` while preserving
the underlying robust space-time planner.

| Pedestrians | Method | Success | Collision | SPL | Min human distance (m) | Social violation rate |
|---:|---|---:|---:|---:|---:|---:|
| 3 | Robust Space-Time Social | 81% | 18% | 0.671 | 0.502 | 0.351 |
| 3 | **Shielded Space-Time Social** | **84%** | **0%** | **0.678** | **0.644** | **0.102** |
| 5 | Robust Space-Time Social | 38% | 23% | 0.303 | 0.486 | 0.452 |
| 5 | **Shielded Space-Time Social** | **47%** | **1%** | **0.344** | **0.648** | **0.104** |
| 10 | Robust Space-Time Social | 11% | 43% | 0.089 | 0.397 | 0.360 |
| 10 | **Shielded Space-Time Social** | **18%** | **7%** | **0.123** | **0.549** | **0.117** |

Collision reduction is substantial, but dense-task completion remains
difficult: Shielded Space-Time Social succeeds in only **18%** of the frozen
`N=10` episodes.

## Navigation methods

The method progression is concise by design:

`A*` → dynamic avoidance → social-aware A* → online replanning and recovery →
predictive social planning → Space-Time Social → Robust Space-Time Social →
multi-pedestrian evaluation → predictive local safety shield

The strongest completed method is `social_spacetime_shielded`. It combines
robust space-time planning with predictive local MOVE/WAIT safety checks,
deterministic physical overrides, and immediate post-override replanning. The
benchmark also retains the earlier static, reactive, social-cost, predictive,
and space-time baselines for controlled comparisons.

## Benchmark design

- Deterministic, seeded scenario generation keeps each method comparison on the
  same initial conditions.
- The density runner supports `N=1`, `N=3`, `N=5`, and `N=10`; the frozen
  headline Robust-versus-Shielded comparison covers `N=3`, `N=5`, and `N=10`.
- Each headline row aggregates 100 diverse episodes generated with seed 42.
- Planning, continuous execution, local safety behavior, and failure evidence
  are evaluated through the same PyBullet simulation pipeline.

The algorithms, scenario semantics, metrics, timeouts, social weights, robot
speed, benchmark results, and demo trajectories are frozen for v1.0.0.

## Metrics

SocialNav-Bench reports:

- success rate and collision rate
- path length and time to goal
- Success weighted by Path Length (SPL)
- minimum human distance
- social violation rate

## Benchmark trends

These figures are generated from the same frozen 100-episode, seed-42 density
comparison. They show the safety improvement alongside the decline in task
completion and SPL as pedestrian density increases.

![Success and collision rates versus pedestrian density](docs/assets/success_collision_vs_density.png)

![Social metrics versus pedestrian density](docs/assets/social_metrics_vs_density.png)

![SPL versus pedestrian density](docs/assets/spl_vs_density.png)

## Failure analysis

Failure analysis is a first-class benchmark output, not an afterthought. The
runtime `planner_failure_reason` records the immediate planner-level outcome;
the post-run `diagnostic_failure_category` records the evidence-based diagnosis.
Some residual failures originally surfaced as `time_horizon_exhausted`, while
diagnostics showed that certain cases were actually
`persistent_dynamic_blockage`: terminal stationary pedestrians had formed
multi-human cut sets that continued to block every feasible route.

Keeping those two fields separate avoids treating a planning symptom as the
underlying environmental cause.

## Reproduce the results

Run all commands from the repository root. Create an isolated Python 3.10+
environment, then install the tracked dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Debian or Ubuntu, install the `python3-venv` operating-system package first
if `ensurepip` is unavailable.

Run the test suite:

```bash
python -m pytest -q
```

Run the complete supported density sweep:

```bash
python experiments/run_density_benchmark.py \
  --episodes 100 \
  --seed 42 \
  --pedestrians 1,3,5,10
```

To reproduce only the headline comparison without overwriting the frozen
artifact, use a separate output path:

```bash
python experiments/run_density_benchmark.py \
  --episodes 100 \
  --seed 42 \
  --pedestrians 3,5,10 \
  --methods social_spacetime_robust,social_spacetime_shielded \
  --output outputs/reproduced_shield_density_benchmark_seed42.json
```

Regenerate the benchmark plots from that JSON:

```bash
python experiments/plot_benchmark_results.py \
  --input outputs/reproduced_shield_density_benchmark_seed42.json \
  --output-dir docs/assets
```

Reproduce the representative demo, including the animation and trajectory
comparison:

```bash
python experiments/render_demo.py \
  --scenario-id diverse-seed-42-episode-0024-pedestrians-3
```

The renderer uses a system `ffmpeg` when available and otherwise uses the
`imageio-ffmpeg` binary installed through `requirements.txt`. Generated JSON is
written under the gitignored `outputs/` directory; the release figures and demo
assets are tracked under `docs/assets/`.

## Project structure

```text
socialnav-bench/
├── socialnav/
│   ├── benchmark/   # scenarios, runners, aggregation, and diagnostics
│   ├── planners/    # classical, social, predictive, space-time, and shield logic
│   ├── env/         # PyBullet environment and pedestrian simulation
│   ├── evaluation/  # episode-level evaluation
│   └── metrics/     # navigation and social metrics
├── experiments/     # benchmark, analysis, plotting, and demo entry points
├── tests/           # regression and behavior tests
├── docs/assets/     # tracked figures, GIF, MP4, and trajectory comparison
├── requirements.txt
└── LICENSE
```

## Limitations

- Evaluation is simulation-only, with no real-robot validation.
- Pedestrian motion is simplified and deterministic.
- Pedestrians stop indefinitely at their terminal targets.
- Stationary terminal pedestrians can create persistent dynamic cut-set
  blockages involving multiple humans.
- High-density multi-human task completion remains difficult; the frozen
  `N=10` Shielded success rate is only 18%.
- The benchmark makes no claim of outperforming all canonical
  social-navigation systems.

## License

SocialNav-Bench is available under the [MIT License](LICENSE).
Copyright (c) 2026 Zhihao Chen.
