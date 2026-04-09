# Lightweight baseline comparison (Codex-web scale)

## Scope and framing
- This is a lightweight local comparison intended as a sanity-but-nontrivial check.
- Results are preliminary and are **not** a final paper-grade benchmark.
- A larger multi-family evaluation on stronger compute should still be run later.

## Exact command run
- `python scripts/run_lightweight_baseline_comparison.py --capacities 3,5,8 --max-requests 200 --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 --out-dir analysis/lightweight_comparison`

## Policies requested
- `blind_oracle`
- `blind_oracle_lru_combiner`
- `evict_value_v1`
- `lru`
- `predictive_marker`
- `rest_v1`
- `trust_and_doubt`

## Traces and settings
- Traces used (6): `file::example_unweighted`, `file::example_atlas_v1`, `synthetic::hot_loop`, `synthetic::bursty_scan`, `synthetic::phase_shift`, `synthetic::mixed_locality`
- Capacities: 3, 5, 8
- Max requests per trace: 200

## Aggregate policy summary
| policy | mean_misses | mean_hit_rate | mean_rel_impr_vs_lru | avg_rank | worst_rank | mean_regret_vs_best | mean_norm_gap | W/T/L vs LRU |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| blind_oracle | 36.722 | 60.110% | 20.621% | 1.11 | 2 | 0.222 | 0.003 | 12/6/0 |
| blind_oracle_lru_combiner | 36.833 | 59.523% | 19.649% | 1.39 | 4 | 0.333 | 0.015 | 11/7/0 |
| evict_value_v1 | 70.222 | 36.745% | -31.965% | 4.56 | 7 | 33.722 | 0.792 | 2/9/7 |
| lru | 54.667 | 46.469% | 0.000% | 4.06 | 7 | 18.167 | 0.368 | 0/18/0 |
| predictive_marker | 45.222 | 53.855% | 12.124% | 2.78 | 5 | 8.722 | 0.136 | 10/8/0 |
| rest_v1 | 48.167 | 51.799% | 8.176% | 2.61 | 5 | 11.667 | 0.237 | 10/8/0 |
| trust_and_doubt | 39.833 | 57.145% | 17.112% | 1.94 | 4 | 3.333 | 0.051 | 10/8/0 |

## evict_value_v1 scorer mode usage
- `lightweight`: 18 run(s)

## Per-trace winners (lower mean misses across capacities)
- `file::example_atlas_v1`: `blind_oracle`, `blind_oracle_lru_combiner`, `rest_v1`
- `file::example_unweighted`: `blind_oracle`, `predictive_marker`, `trust_and_doubt`
- `synthetic::bursty_scan`: `blind_oracle`, `blind_oracle_lru_combiner`
- `synthetic::hot_loop`: `trust_and_doubt`
- `synthetic::mixed_locality`: `blind_oracle`
- `synthetic::phase_shift`: `blind_oracle`, `blind_oracle_lru_combiner`

## Main caveats
- Synthetic traces are included to provide lightweight diversity because repository examples are very short.
- Prediction metadata is generated heuristically from each trace sequence; this is suitable for quick local checks, not definitive benchmarking.
- `evict_value_v1` is included via a text-only lightweight surrogate scorer in this run; interpret this as local-surrogate behavior, not trained-checkpoint performance.
