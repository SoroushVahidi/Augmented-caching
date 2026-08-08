# KBS reviewer-expanded policy-comparison merge report

This is a **draft / available-capacities** merge, not the canonical full heavy-r1 artifact.
The canonical filename `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` remains absent by design until cap256 is completed or the manuscript scope is explicitly changed.

## Scope
- Input chunks: analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv, analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv, analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv
- Capacities covered: 32, 64, 128
- Cap256: not run, therefore excluded
- Policy set: `lru`, `sieve`, `fifo_reinsertion`, `predictive_marker`, `blind_oracle_lru_combiner`, `trust_and_doubt`, `rest_v1`, `evict_value_v1`
- Historical bare `cap32.csv`: not merged, because it uses a different 7-policy roster and includes `blind_oracle` instead of `sieve` / `fifo_reinsertion`.

## Merge Result
- Row count: 168
- Capacities covered: 32, 64, 128
- Trace families covered: brightkite, citibike, cloudphysics, metacdn, metakv, twemcache, wiki2018
- Policies covered: blind_oracle_lru_combiner, evict_value_v1, fifo_reinsertion, lru, predictive_marker, rest_v1, sieve, trust_and_doubt

## Per-policy Means
| policy | mean misses | mean hit rate |
|---|---:|---:|
| blind_oracle_lru_combiner | 33605.5714 | 0.327889 |
| evict_value_v1 | 35689.9048 | 0.286202 |
| fifo_reinsertion | 33630.2381 | 0.327395 |
| lru | 33604.2857 | 0.327914 |
| predictive_marker | 33760.5238 | 0.324790 |
| rest_v1 | 33604.2857 | 0.327914 |
| sieve | 34318.5238 | 0.313630 |
| trust_and_doubt | 34004.8571 | 0.319903 |

## Per-family Mean Misses
| trace_family | evict_value_v1 | sieve | fifo_reinsertion | lru | rest_v1 |
|---|---:|---:|---:|---:|---:|
| brightkite | 20548.6667 | 16945.3333 | 16734.3333 | 16700.0000 | 16700.0000 |
| citibike | 22114.0000 | 19316.0000 | 18630.3333 | 18694.0000 | 18694.0000 |
| cloudphysics | 49398.3333 | 48584.0000 | 48465.0000 | 48532.0000 | 48532.0000 |
| metacdn | 32686.3333 | 28755.3333 | 28554.0000 | 28602.3333 | 28602.3333 |
| metakv | 38342.3333 | 38000.6667 | 38014.6667 | 38029.6667 | 38029.6667 |
| twemcache | 36739.6667 | 38628.3333 | 35013.3333 | 34672.0000 | 34672.0000 |
| wiki2018 | 50000.0000 | 50000.0000 | 50000.0000 | 50000.0000 | 50000.0000 |

## Average Comparison vs LRU
- evict_value_v1: does not beat LRU (35689.9048 vs 33604.2857 mean misses)
- sieve: does not beat LRU (34318.5238 vs 33604.2857 mean misses)
- fifo_reinsertion: does not beat LRU (33630.2381 vs 33604.2857 mean misses)
- rest_v1: ties LRU (33604.2857 vs 33604.2857 mean misses)

## Canonicality
- Safe for manuscript drafting: **yes**, but only as the draft available-capacities artifact.
- Safe for final canonical KBS claims: **no**; cap256 is still absent, so the full canonical heavy-r1 bundle is incomplete.
- Recommended canonical file policy: keep `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` absent until cap256 is completed or the scope is explicitly changed.

