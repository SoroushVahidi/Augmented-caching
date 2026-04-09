# Offline-teacher vs heuristic supervision (medium-scale local evaluation)

## Configuration
- model family: `random_forest`
- traces: `data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json` (4 resolved traces)
- capacities: `[2, 3, 4, 5]`
- horizon: `16`
- seeds: `[0, 1, 2, 3, 4, 5]`

## Disagreement
- decision disagreement rate: **0.4250** (17 / 40)
- mean gain on disagreement decisions: **0.0370**
- mean gain on agreement decisions: **-0.0606**

## Downstream aggregate
| label_source | workloads | total_misses | hit_rate | delta_misses_vs_lru | avg_rank |
|---|---:|---:|---:|---:|---:|
| heuristic | 20 | 116 | 0.4200 | -10 | 1.0000 |
| offline_teacher | 20 | 118 | 0.4100 | -8 | 2.0000 |

## Wins / ties / losses
- offline_teacher wins: **0**
- ties: **18**
- offline_teacher losses: **2**

## Recommendation
- Evidence does not yet justify replacing heuristic labels; keep heuristic default and continue targeted offline_teacher experiments.
