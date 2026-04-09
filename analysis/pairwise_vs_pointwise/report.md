# Pairwise vs Pointwise supervision experiment

## Setup
- traces: `data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json` (4 traces)
- capacities: `[2, 3, 4, 5]`
- horizon: `16`
- seeds: `[0, 1, 2, 3, 4, 5]`
- pointwise model: `ridge_regression_with_standard_scaler`
- pairwise model: `logistic_regression_on_delta_features`

## Downstream aggregate
| style | workloads | total_misses | hit_rate | delta_vs_lru | avg_rank |
|---|---:|---:|---:|---:|---:|
| pointwise | 20 | 104 | 0.4800 | -22 | 1.0000 |
| pairwise | 20 | 106 | 0.4700 | -20 | 1.1000 |

- Pairwise wins/ties/losses vs pointwise: **0/18/2**
- Pairwise does not yet beat pointwise downstream; keep pointwise as default while iterating pairwise design.
