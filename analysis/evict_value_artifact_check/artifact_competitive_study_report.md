# artifact-backed competitive study for evict_value_v1 (controlled moderate run)

## Study scope
- Training data for artifact model: full available horizon-8 rows in repo-derived dataset (train=51, val=50).
- Eval grid: 5 traces × 2 regimes × 3 capacities = 30 slices.
- Policies compared: artifact-backed evict_value_v1, lightweight evict_value_v1, robust_ftp_d_marker, blind_oracle_lru_combiner, rest_v1, atlas_v3, lru.
- No persistent binary artifact was stored in-repo; artifact checkpoint was temporary for runtime only.

## Aggregate means (misses lower is better)
| policy | runs | mean_misses | mean_hit_rate |
|---|---:|---:|---:|
| atlas_v3 | 30 | 6.067 | 0.393 |
| blind_oracle_lru_combiner | 30 | 5.900 | 0.410 |
| evict_value_v1_artifact | 30 | 5.900 | 0.410 |
| evict_value_v1_lightweight | 30 | 7.600 | 0.240 |
| lru | 30 | 6.933 | 0.307 |
| rest_v1 | 30 | 6.333 | 0.367 |
| robust_ftp_d_marker | 30 | 5.767 | 0.423 |

## Direct competitiveness of artifact-backed evict_value_v1
| comparator | mean miss delta (artifact - comparator) | W/T/L | best delta | worst delta |
|---|---:|---:|---:|---:|
| evict_value_v1_lightweight | -1.700 | 24/6/0 | -4 | 0 |
| robust_ftp_d_marker | 0.133 | 0/26/4 | 0 | 1 |
| blind_oracle_lru_combiner | 0.000 | 3/23/4 | -2 | 1 |
| rest_v1 | -0.433 | 10/19/1 | -2 | 1 |
| atlas_v3 | -0.167 | 6/23/1 | -1 | 1 |
| lru | -1.033 | 18/12/0 | -3 | 0 |

## Requested explicit answers
1. **Is artifact-backed evict_value_v1 clearly better than lightweight?**
   - Yes on this controlled run: artifact beats lightweight in 24/30 slices (ties 6, losses 0) with mean delta -1.700 misses.
2. **Is it competitive with strong robust baselines?**
   - Partially competitive, mostly via ties: vs robust_ftp_d_marker W/T/L=0/26/4; vs blind_oracle_lru_combiner W/T/L=3/23/4; vs atlas_v3 W/T/L=6/23/1.
   - It is not clearly stronger than the top robust baselines in this moderate check.
3. **In what slices does it win, tie, or lose?**
   - Detailed per-slice outcomes are in `artifact_competitive_study_summary.json` (`slice_breakdown`).
   - Worst losses are bounded (max +1 miss vs strong robust baselines), and many outcomes are ties.
4. **Should evict_value remain main, secondary, or de-prioritized?**
   - **secondary line** (conservative).

## Notes
- This remains a moderate synthetic/lightweight harness study; not a full production-scale claim.
- Next escalation (if needed) should increase trace diversity and request depth before changing portfolio priority.
