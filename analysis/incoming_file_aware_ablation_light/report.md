# Incoming-file-aware lightweight ablation for `evict_value_v1`

## Scope and isolation
- Experimental path only (`src/lafc/experiments/`, `scripts/experiments/`, and this analysis directory).
- Uses existing repository traces only (no new datasets).
- Keeps model family fixed: `HistGradientBoostingRegressor` for both scorers.
- Does not invoke or modify the canonical heavy_r1 manuscript pipeline.

## Existing baseline pipeline identified
- Candidate-level builder: `src/lafc/evict_value_dataset_v1.py::build_evict_value_examples_v1`.
- Feature pipeline: `src/lafc/evict_value_features_v1.py::compute_candidate_features_v1` and `EVICT_VALUE_V1_FEATURE_COLUMNS`.
- Canonical lightweight train script: `scripts/train_evict_value_v1.py`.

## Incoming-file-aware additions
- Added incoming/current-request relation features:
  - `incoming_recent_request_rate`
  - `incoming_recent_hit_rate`
  - `incoming_last_seen_gap_norm`
  - `incoming_candidate_transition_rate`
  - `incoming_bucket_gap_to_candidate`
  - `incoming_confidence_gap_to_candidate`
  - `incoming_bucket_matches_candidate`

## Ranking quality (candidate-level)
- Base test top1 match: 0.5000
- Incoming-aware test top1 match: 0.5000
- Base test mean regret: 0.0000
- Incoming-aware test mean regret: 0.0000

## Downstream replay misses
- Mean base misses: 5.556
- Mean incoming-aware misses: 5.556
- Mean miss delta (base - incoming-aware): 0.000

## Interpretation
- In this lightweight subset run, explicit incoming-file conditioning had neutral average replay misses.
- Treat this as a controlled lightweight signal, not a heavy_r1 manuscript claim.
