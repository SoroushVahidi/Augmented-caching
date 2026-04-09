# evict_value_v1 tiny artifact-backed check

## Scope (minimal)
- Tiny training subset: 51 rows (horizon=8) from `data/derived/evict_value_v1_train.csv`.
- Tiny validation subset: 50 rows from `data/derived/evict_value_v1_val.csv`.
- Tiny eval grid: 4 traces × 2 regimes × 2 capacities = 16 slices.
- Compared policies: artifact-backed `evict_value_v1`, lightweight `evict_value_v1`, `robust_ftp_d_marker`, `blind_oracle_lru_combiner`, `rest_v1`, `lru`.
- No persistent binary output was kept; a temporary artifact file was created in a temp directory only for runtime loading and then deleted automatically.

## Headline results
- Tiny artifact model validation MAE: **1.669**.
- Artifact vs lightweight mean miss delta (artifact - lightweight): **-2.312**.
- Artifact better/tie/worse vs lightweight: **16/0/0** slices.

## Pairwise vs required baselines (artifact-backed evict_value_v1)
| comparator | mean miss delta (artifact - comparator) | W/T/L | worst delta |
|---|---:|---:|---:|
| robust_ftp_d_marker | 0.188 | 0/13/3 | 1 |
| blind_oracle_lru_combiner | 0.062 | 2/11/3 | 1 |
| rest_v1 | -0.625 | 8/7/1 | 1 |
| lru | -1.312 | 12/4/0 | 0 |

## Conservative answers
1. **Does artifact-backed mode materially change performance?**
   - It improves over lightweight on some slices, but not enough to make evict_value competitive with the strongest baselines in this tiny check.
2. **Was lightweight surrogate misleading?**
   - Partially yes: artifact outperforms lightweight on multiple slices, so surrogate-only verdicts were somewhat pessimistic.
3. **Should evict_value stay alive as a main research direction?**
   - **continue only behind artifact-backed checks (not as main line)** based on this tiny evidence.
