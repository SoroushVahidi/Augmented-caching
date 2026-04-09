# Method-family winner selection (lightweight)

## Scope
- Lightweight cross-family comparison on repo examples + tiny stress traces.
- Uses two prediction-quality regimes: `clean` and `noisy`.
- Intended for ranking guidance, not paper-grade benchmarking.

## Command
- `python scripts/run_family_winner_selection.py --capacities 2,3,4 --max-requests 120 --regimes clean,noisy --ml-gate-horizon 8 --ml-gate-model-path analysis/family_winner_selection/models/ml_gate_v2_lightweight.pkl --out-dir analysis/family_winner_selection`

## Aggregate ranking
| policy | runs | mean_misses | mean_hit_rate | avg_rank | mean_rel_impr_vs_lru | W/T/L vs LRU |
|---|---:|---:|---:|---:|---:|---|
| robust_ftp_d_marker | 30 | 5.767 | 42.333% | 1.07 | 13.889% | 18/12/0 |
| blind_oracle_lru_combiner | 30 | 5.900 | 41.000% | 1.47 | 11.905% | 15/15/0 |
| trust_and_doubt | 30 | 5.967 | 40.333% | 1.63 | 11.852% | 18/12/0 |
| atlas_v3 | 30 | 6.067 | 39.333% | 1.77 | 10.460% | 17/13/0 |
| ml_gate_v2 | 30 | 6.133 | 38.667% | 1.87 | 10.042% | 18/12/0 |
| rest_v1 | 30 | 6.333 | 36.667% | 2.50 | 7.349% | 11/19/0 |
| lru | 30 | 6.933 | 30.667% | 4.23 | 0.000% | 0/30/0 |
| evict_value_v1 | 30 | 7.600 | 24.000% | 6.10 | -10.688% | 0/17/13 |

## Verdict
- Strongest family candidate in this lightweight run: **robust_ftp_d_marker**.
- Unpromising family candidate in this lightweight run: **evict_value_v1**.
- Treat this as directional only; verify on larger traces before making durable claims.
