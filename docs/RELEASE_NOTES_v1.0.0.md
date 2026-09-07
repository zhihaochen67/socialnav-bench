# SocialNav-Bench v1.0.0

First stable release of SocialNav-Bench.

## Highlights

- Reproducible, seeded social-navigation benchmark
- Static, dynamic, and social-aware planning baselines
- Predictive social and explicit space-time planning
- Robust continuous execution with replanning
- Multi-pedestrian evaluation at increasing densities
- Predictive local MOVE/WAIT safety shield
- Failure taxonomy with persistent-blockage diagnostics
- Benchmark visualizations and a reproducible representative demo

## Key results

The frozen Phase 7D comparison evaluates 100 diverse episodes per density with
seed 42. Shielding reduces collision rate from 18% to 0% at `N=3`, 23% to 1%
at `N=5`, and 43% to 7% at `N=10`.

| Pedestrians | Robust success | Robust collision | Shielded success | Shielded collision | Shielded min human distance (m) | Shielded social violation rate |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 81% | 18% | 84% | 0% | 0.644 | 0.102 |
| 5 | 38% | 23% | 47% | 1% | 0.648 | 0.104 |
| 10 | 11% | 43% | 18% | 7% | 0.549 | 0.117 |

The collision improvement does not solve dense navigation: Shielded Space-Time
Social succeeds in only 18% of the frozen `N=10` episodes.

## Reproducibility

- Dependencies are constrained by compatible ranges in `requirements.txt`.
- The complete test suite runs with `python -m pytest -q`.
- Benchmarks use deterministic seeds and fixed scenario-generation semantics.
- `experiments/run_density_benchmark.py` reproduces density sweeps.
- `experiments/plot_benchmark_results.py` regenerates benchmark figures from
  benchmark JSON.
- `experiments/render_demo.py` replays the frozen representative scenario and
  renders the GIF, MP4, and trajectory comparison.

## Limitations

- Evaluation is simulation-only; no real-robot validation has been performed.
- Pedestrian motion is simplified, and pedestrians stop indefinitely at their
  terminal targets.
- Terminal pedestrians can form persistent multi-human dynamic cut-set
  blockages.
- High-density task completion remains difficult, including only 18% Shielded
  success at `N=10`.
- No claim is made that this benchmark outperforms all canonical
  social-navigation systems.
