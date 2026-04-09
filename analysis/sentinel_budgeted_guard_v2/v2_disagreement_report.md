# sentinel_budgeted_guard_v2 disagreement-stress report

## Setup
- Suite: synthetic disagreement stress (help/hurt/mixed traces).
- Capacities: [2, 3]
- Policies compared: sentinel_budgeted_guard_v2, sentinel_robust_tripwire_v1, robust_ftp_d_marker, blind_oracle_lru_combiner, lru.

## Aggregate
- v2 vs v1 (W/T/L): 1/2/3
- Mean miss delta (v2 - v1): 0.333
- Mean miss delta (v2 - robust_ftp_d_marker): -0.167
- Harmful overrides (v2 vs v1): 2 vs 0

## Explicit answers
- **Does v2 improve over sentinel_robust_tripwire_v1 on disagreement slices?** No.
- **Does it introduce more harmful overrides?** Yes.
- **Does it remain robustness-preserving?** Yes (mean v2-robust delta=-0.167).
- **Should v2 replace v1 as the main candidate line?** No.

## Per-slice
| trace | case | cap | disagreement_steps | v2_vs_v1 | v2_minus_v1 | v2_harmful | v1_harmful |
|---|---|---:|---:|---|---:|---:|---:|
| synthetic::predictor_disagreement_help | predictor_help | 2 | 4 | tie | 0 | 0 | 0 |
| synthetic::predictor_disagreement_help | predictor_help | 3 | 4 | loss | 1 | 1 | 0 |
| synthetic::predictor_disagreement_hurt | predictor_hurt | 2 | 9 | tie | 0 | 1 | 0 |
| synthetic::predictor_disagreement_hurt | predictor_hurt | 3 | 2 | win | -1 | 0 | 0 |
| synthetic::predictor_disagreement_mixed | mixed | 2 | 4 | loss | 1 | 0 | 0 |
| synthetic::predictor_disagreement_mixed | mixed | 3 | 3 | loss | 1 | 0 | 0 |
