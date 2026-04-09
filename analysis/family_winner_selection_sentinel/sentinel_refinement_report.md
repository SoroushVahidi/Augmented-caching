# Sentinel refinement report (focused pass)

## Refinement implemented
- Single targeted refinement: predictor override is now allowed **only on disagreement steps** where robust and predictor shadows propose different evictions.
- Rationale: avoid spending override budget on agreement steps that cannot change outcomes.

## Lightweight comparison outcome (same harness)
| comparator | W/T/L (sentinel) | mean miss delta (sentinel - comparator) |
|---|---:|---:|
| robust_ftp_d_marker | 0/30/0 | 0.000 |
| blind_oracle_lru_combiner | 3/27/0 | -0.133 |
| rest_v1 | 11/18/1 | -0.567 |
| atlas_v3 | 8/21/1 | -0.300 |
| lru | 18/12/0 | -1.167 |

## Explicit answers
- **Did the refinement create any wins over robust_ftp_d_marker?** No (wins=0).
- **Did it introduce any losses vs robust_ftp_d_marker?** No (losses=0).
- **Should refined sentinel remain the main candidate line?** Yes: keep as candidate main line needing refinement, because robustness parity is preserved but no separation from robust_ftp_d_marker appears yet.

## Override activity check
- Mean predictor coverage: 0.000 (min=0.000, max=0.000).
- Mean predictor override steps: 0.00 across 30 slices.
- Guard triggers: mean 0.00; nonzero on 0/30 slices.
- Observation: override remains conservative and still tracks the robust winner closely on this lightweight suite.

## Exact slices
- Full per-slice outcomes are in `sentinel_refinement_slice_breakdown.csv`.
