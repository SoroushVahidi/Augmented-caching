# Sentinel comparison report (lightweight winner-selection)

## Scope and caution
- Dataset/slices: same lightweight family harness slices (5 traces × 2 regimes × 3 capacities = 30 slices).
- This is directional evidence only; not a heavy benchmark.

## 1) Headline comparison vs requested baselines
| comparator | W/T/L (sentinel) | mean miss delta (sentinel - comparator) |
|---|---:|---:|
| robust_ftp_d_marker | 0/30/0 | 0.000 |
| blind_oracle_lru_combiner | 3/27/0 | -0.133 |
| rest_v1 | 11/18/1 | -0.567 |
| atlas_v3 | 8/21/1 | -0.300 |
| lru | 18/12/0 | -1.167 |

Interpretation: negative mean delta is better for sentinel.

## 2) Exact slices of wins/ties/losses
### vs robust_ftp_d_marker
- Wins (0):
  - None
- Ties (30):
  - file::example_atlas_v1 | clean | k=2 (delta=+0)
  - file::example_atlas_v1 | clean | k=3 (delta=+0)
  - file::example_atlas_v1 | clean | k=4 (delta=+0)
  - file::example_atlas_v1 | noisy | k=2 (delta=+0)
  - file::example_atlas_v1 | noisy | k=3 (delta=+0)
  - file::example_atlas_v1 | noisy | k=4 (delta=+0)
  - file::example_unweighted | clean | k=2 (delta=+0)
  - file::example_unweighted | clean | k=3 (delta=+0)
  - file::example_unweighted | clean | k=4 (delta=+0)
  - file::example_unweighted | noisy | k=2 (delta=+0)
  - file::example_unweighted | noisy | k=3 (delta=+0)
  - file::example_unweighted | noisy | k=4 (delta=+0)
  - stress::mixed_regime | clean | k=2 (delta=+0)
  - stress::mixed_regime | clean | k=3 (delta=+0)
  - stress::mixed_regime | clean | k=4 (delta=+0)
  - stress::mixed_regime | noisy | k=2 (delta=+0)
  - stress::mixed_regime | noisy | k=3 (delta=+0)
  - stress::mixed_regime | noisy | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | clean | k=2 (delta=+0)
  - stress::predictor_good_lru_bad | clean | k=3 (delta=+0)
  - stress::predictor_good_lru_bad | clean | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | noisy | k=2 (delta=+0)
  - stress::predictor_good_lru_bad | noisy | k=3 (delta=+0)
  - stress::predictor_good_lru_bad | noisy | k=4 (delta=+0)
- Losses (0):
  - None

### vs blind_oracle_lru_combiner
- Wins (3):
  - file::example_unweighted | noisy | k=3 (delta=-1)
  - stress::mixed_regime | noisy | k=3 (delta=-2)
  - stress::predictor_good_lru_bad | noisy | k=3 (delta=-1)
- Ties (27):
  - file::example_atlas_v1 | clean | k=2 (delta=+0)
  - file::example_atlas_v1 | clean | k=3 (delta=+0)
  - file::example_atlas_v1 | clean | k=4 (delta=+0)
  - file::example_atlas_v1 | noisy | k=2 (delta=+0)
  - file::example_atlas_v1 | noisy | k=3 (delta=+0)
  - file::example_atlas_v1 | noisy | k=4 (delta=+0)
  - file::example_unweighted | clean | k=2 (delta=+0)
  - file::example_unweighted | clean | k=3 (delta=+0)
  - file::example_unweighted | clean | k=4 (delta=+0)
  - file::example_unweighted | noisy | k=2 (delta=+0)
  - file::example_unweighted | noisy | k=4 (delta=+0)
  - stress::mixed_regime | clean | k=2 (delta=+0)
  - stress::mixed_regime | clean | k=3 (delta=+0)
  - stress::mixed_regime | clean | k=4 (delta=+0)
  - stress::mixed_regime | noisy | k=2 (delta=+0)
  - stress::mixed_regime | noisy | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | clean | k=2 (delta=+0)
  - stress::predictor_good_lru_bad | clean | k=3 (delta=+0)
  - stress::predictor_good_lru_bad | clean | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | noisy | k=2 (delta=+0)
  - stress::predictor_good_lru_bad | noisy | k=4 (delta=+0)
- Losses (0):
  - None

### vs rest_v1
- Wins (11):
  - file::example_atlas_v1 | clean | k=2 (delta=-2)
  - file::example_atlas_v1 | noisy | k=2 (delta=-1)
  - file::example_unweighted | clean | k=2 (delta=-3)
  - file::example_unweighted | clean | k=3 (delta=-1)
  - file::example_unweighted | noisy | k=2 (delta=-2)
  - file::example_unweighted | noisy | k=3 (delta=-1)
  - stress::mixed_regime | clean | k=2 (delta=-3)
  - stress::mixed_regime | noisy | k=2 (delta=-1)
  - stress::predictor_good_lru_bad | clean | k=2 (delta=-1)
  - stress::predictor_good_lru_bad | clean | k=3 (delta=-2)
  - stress::predictor_good_lru_bad | noisy | k=3 (delta=-1)
- Ties (18):
  - file::example_atlas_v1 | clean | k=3 (delta=+0)
  - file::example_atlas_v1 | clean | k=4 (delta=+0)
  - file::example_atlas_v1 | noisy | k=3 (delta=+0)
  - file::example_atlas_v1 | noisy | k=4 (delta=+0)
  - file::example_unweighted | clean | k=4 (delta=+0)
  - file::example_unweighted | noisy | k=4 (delta=+0)
  - stress::mixed_regime | clean | k=3 (delta=+0)
  - stress::mixed_regime | clean | k=4 (delta=+0)
  - stress::mixed_regime | noisy | k=3 (delta=+0)
  - stress::mixed_regime | noisy | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | clean | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | noisy | k=4 (delta=+0)
- Losses (1):
  - stress::predictor_good_lru_bad | noisy | k=2 (delta=+1)

### vs atlas_v3
- Wins (8):
  - file::example_atlas_v1 | clean | k=2 (delta=-1)
  - file::example_unweighted | clean | k=2 (delta=-2)
  - file::example_unweighted | noisy | k=2 (delta=-1)
  - stress::mixed_regime | clean | k=2 (delta=-2)
  - stress::mixed_regime | noisy | k=3 (delta=-1)
  - stress::predictor_good_lru_bad | clean | k=2 (delta=-1)
  - stress::predictor_good_lru_bad | clean | k=3 (delta=-1)
  - stress::predictor_good_lru_bad | noisy | k=3 (delta=-1)
- Ties (21):
  - file::example_atlas_v1 | clean | k=3 (delta=+0)
  - file::example_atlas_v1 | clean | k=4 (delta=+0)
  - file::example_atlas_v1 | noisy | k=2 (delta=+0)
  - file::example_atlas_v1 | noisy | k=3 (delta=+0)
  - file::example_atlas_v1 | noisy | k=4 (delta=+0)
  - file::example_unweighted | clean | k=3 (delta=+0)
  - file::example_unweighted | clean | k=4 (delta=+0)
  - file::example_unweighted | noisy | k=3 (delta=+0)
  - file::example_unweighted | noisy | k=4 (delta=+0)
  - stress::mixed_regime | clean | k=3 (delta=+0)
  - stress::mixed_regime | clean | k=4 (delta=+0)
  - stress::mixed_regime | noisy | k=2 (delta=+0)
  - stress::mixed_regime | noisy | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | clean | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | noisy | k=4 (delta=+0)
- Losses (1):
  - stress::predictor_good_lru_bad | noisy | k=2 (delta=+1)

### vs lru
- Wins (18):
  - file::example_atlas_v1 | clean | k=2 (delta=-2)
  - file::example_atlas_v1 | clean | k=3 (delta=-3)
  - file::example_atlas_v1 | clean | k=4 (delta=-1)
  - file::example_atlas_v1 | noisy | k=2 (delta=-2)
  - file::example_atlas_v1 | noisy | k=3 (delta=-3)
  - file::example_atlas_v1 | noisy | k=4 (delta=-1)
  - file::example_unweighted | clean | k=2 (delta=-3)
  - file::example_unweighted | clean | k=3 (delta=-1)
  - file::example_unweighted | noisy | k=2 (delta=-3)
  - file::example_unweighted | noisy | k=3 (delta=-1)
  - stress::mixed_regime | clean | k=2 (delta=-3)
  - stress::mixed_regime | clean | k=3 (delta=-2)
  - stress::mixed_regime | noisy | k=2 (delta=-2)
  - stress::mixed_regime | noisy | k=3 (delta=-2)
  - stress::predictor_good_lru_bad | clean | k=2 (delta=-2)
  - stress::predictor_good_lru_bad | clean | k=3 (delta=-2)
  - stress::predictor_good_lru_bad | noisy | k=2 (delta=-1)
  - stress::predictor_good_lru_bad | noisy | k=3 (delta=-1)
- Ties (12):
  - file::example_unweighted | clean | k=4 (delta=+0)
  - file::example_unweighted | noisy | k=4 (delta=+0)
  - stress::mixed_regime | clean | k=4 (delta=+0)
  - stress::mixed_regime | noisy | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | clean | k=4 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=2 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=3 (delta=+0)
  - stress::predictor_bad_lru_good | noisy | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | clean | k=4 (delta=+0)
  - stress::predictor_good_lru_bad | noisy | k=4 (delta=+0)
- Losses (0):
  - None

## 3) Is learned override active enough to matter?
- Mean predictor override coverage: 0.200 (min=0.200, max=0.200).
- Mean predictor override steps per slice: 2.00.
- Slices with non-zero override: 30/30.
- Mean guard triggers per slice: 0.00; slices with triggers: 0/30.
- Reading: override is active but conservative; behavior often tracks robust baseline closely.

## 4) Any gains without obvious robustness loss?
- Vs LRU: non-inferior on 30/30 slices, losses on 0/30.
- Vs robust_ftp_d_marker: losses on 0/30 slices.
- Vs blind_oracle_lru_combiner: losses on 0/30 slices.
- In this run, sentinel exactly matches robust_ftp_d_marker aggregate performance (no clear robustness degradation, but also limited incremental gain over the top robust line).

## 5) Recommendation
- **Keep as candidate main line needing refinement**.
- Reason: robust behavior is preserved and performance is competitive, but evidence here shows little separation from current best robust baseline.
