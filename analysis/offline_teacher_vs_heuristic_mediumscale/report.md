# Offline-teacher vs heuristic supervision (medium-scale local evaluation)

## Configuration
- model family: `ridge`
- traces: `data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json` (4 resolved traces)
- capacities: `[2, 3, 4, 5]`
- horizon: `16`
- seeds: `[0, 1, 2, 3, 4, 5]`

## Disagreement
- decision disagreement rate: **0.4250** (17 / 40)
- mean gain on disagreement decisions: **-0.2222**
- mean gain on agreement decisions: **-0.4545**

## Downstream aggregate
| label_source | workloads | total_misses | hit_rate | delta_misses_vs_lru | avg_rank |
|---|---:|---:|---:|---:|---:|
| heuristic | 20 | 104 | 0.4800 | -22 | 1.0000 |
| offline_teacher | 20 | 122 | 0.3900 | -4 | 2.0000 |

## Wins / ties / losses
- offline_teacher wins: **0**
- ties: **10**
- offline_teacher losses: **10**

## Recommendation
- Evidence does not yet justify replacing heuristic labels; keep heuristic default and continue targeted offline_teacher experiments.
