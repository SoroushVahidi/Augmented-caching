# Sentinel disagreement-stress report

## Setup
- Small synthetic suite purpose-built to induce disagreement between robust and predictor shadows.
- Traces: 3 (help/hurt/mixed), capacities: [2, 3], total slices: 6.
- Policies compared: sentinel_robust_tripwire_v1, robust_ftp_d_marker, blind_oracle_lru_combiner, lru.

## Aggregate results
- Sentinel vs robust_ftp_d_marker (W/T/L): 3/2/1.
- Mean miss delta (sentinel - robust_ftp_d_marker): -0.500.
- Sentinel vs blind_oracle_lru_combiner mean delta: 0.333.
- Sentinel vs lru mean delta: 0.500.

## Explicit answers
- **Does the suite generate disagreement states?** Yes. Disagreement appears on 6/6 slices, mean 4.33 steps per slice.
- **Does sentinel produce wins over robust_ftp_d_marker here?** Yes (3 wins, 1 losses).
- **Does sentinel introduce harmful overrides?** Yes (harmful override steps=2, helpful=5).
- **Is mechanism worth refining further?** Yes, cautiously: disagreement is now observable and override behavior can be studied under stress; continue refinement if wins appear without rising harmful overrides.

## Per-slice table
| trace | case | cap | disagreement_steps | sentinel_vs_robust_ftp_d_marker | sentinel_minus_robust | predictor_steps | helpful_overrides | harmful_overrides |
|---|---|---:|---:|---|---:|---:|---:|---:|
| synthetic::predictor_disagreement_help | predictor_help | 2 | 4 | win | -1 | 3.0 | 1 | 0 |
| synthetic::predictor_disagreement_help | predictor_help | 3 | 4 | tie | 0 | 3.0 | 1 | 1 |
| synthetic::predictor_disagreement_hurt | predictor_hurt | 2 | 9 | loss | 1 | 1.0 | 0 | 1 |
| synthetic::predictor_disagreement_hurt | predictor_hurt | 3 | 2 | tie | 0 | 1.0 | 0 | 0 |
| synthetic::predictor_disagreement_mixed | mixed | 2 | 4 | win | -1 | 2.0 | 1 | 0 |
| synthetic::predictor_disagreement_mixed | mixed | 3 | 3 | win | -2 | 3.0 | 2 | 0 |
