# Sentinel semi-realistic targeted evaluation

## Setup
- Goal: bridge ordinary lightweight suite and fully synthetic disagreement-stress suite.
- Traces are generated with locality drift, short scan bursts, and daypart popularity shifts; predicted caches are recency-frequency based with lag/noise.
- Traces: 3, capacities: [3, 4], slices: 6.
- Policies: sentinel_robust_tripwire_v1, robust_ftp_d_marker, blind_oracle_lru_combiner, rest_v1, atlas_v3, lru.

## Aggregate
- Sentinel vs robust_ftp_d_marker (W/T/L): 2/1/3.
- Mean delta misses (sentinel - robust_ftp_d_marker): 0.333.
- Mean misses by policy: {'sentinel_robust_tripwire_v1': 135.0, 'robust_ftp_d_marker': 134.66666666666666, 'blind_oracle_lru_combiner': 137.66666666666666, 'rest_v1': 136.5, 'atlas_v3': 136.5, 'lru': 136.5}.
- Helpful/harmful overrides: 3/5.

## Explicit answers
- **Does v1 show any real wins in this semi-realistic setting?** Yes (2 wins).
- **Does it preserve robustness?** No (mean sentinel-robust delta=0.333).
- **Is it still the best main empirical candidate in the repo?** No (best mean-miss policy here: robust_ftp_d_marker).

## Per-slice sentinel vs robust
| trace | cap | disagreement_steps | sentinel_vs_robust | sentinel_minus_robust | helpful_overrides | harmful_overrides |
|---|---:|---:|---|---:|---:|---:|
| semi::phase_locality_drift | 3 | 14 | loss | 1 | 0 | 1 |
| semi::phase_locality_drift | 4 | 1 | win | -1 | 1 | 0 |
| semi::locality_with_short_scans | 3 | 63 | loss | 2 | 0 | 2 |
| semi::locality_with_short_scans | 4 | 0 | tie | 0 | 0 | 0 |
| semi::daypart_popularity_mix | 3 | 34 | loss | 1 | 0 | 1 |
| semi::daypart_popularity_mix | 4 | 10 | win | -1 | 2 | 1 |
