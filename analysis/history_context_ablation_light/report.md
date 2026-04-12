# History-context lightweight ablation for `evict_value_v1`

## Scope and isolation
- Separate experimental path only (`src/lafc/experiments/`, `scripts/experiments/`, and this analysis directory).
- Uses only repository-provided compact traces (`repo_light`) and identical capacities/horizon used in the prior lightweight incoming-file ablation.
- Keeps model family fixed for both arms: `HistGradientBoostingRegressor`.
- Does not modify or rerun canonical heavy_r1 manuscript pipelines/artifacts.

## Base pipeline identification
- Base feature arm is exactly `EVICT_VALUE_V1_FEATURE_COLUMNS` from `src/lafc/evict_value_features_v1.py`.
- Base candidate-generation semantics follow the same candidate-level replay setup used by `build_evict_value_examples_v1`.

## Added history-aware features
- `hist_w8_candidate_request_rate`
- `hist_w16_candidate_request_rate`
- `hist_w32_candidate_request_rate`
- `hist_w8_candidate_hit_rate`
- `hist_w16_candidate_hit_rate`
- `hist_candidate_last_seen_gap_norm`
- `hist_candidate_last_hit_gap_norm`
- `hist_candidate_interarrival_mean_norm`
- `hist_candidate_interarrival_std_norm`
- `hist_candidate_burst_max_norm`
- `hist_candidate_recent_trend_w8_minus_w32`
- `hist_global_unique_ratio_w16`
- `hist_global_repeat_rate_w16`
- `hist_global_hit_ratio_w16`
- `hist_transition_prev_to_candidate_rate_w16`

## Candidate-ranking quality
- Base test top1 match: 0.6667
- History-aware test top1 match: 0.6667
- Base test mean regret: 0.0000
- History-aware test mean regret: 0.0000

## Downstream replay misses
- Mean base misses: 5.667
- Mean history-aware misses: 5.500
- Mean miss delta (base - history-aware): 0.167
- Verdict on richer history features in this lightweight run: **helps**.
