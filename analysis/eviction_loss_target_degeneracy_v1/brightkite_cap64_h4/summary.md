# Eviction-Loss Target Degeneracy Diagnostic

Status: `COMPLETE`

This is a read-only mechanism diagnostic. It does not change the deployed policy.

## Event Summary

- `event_count`: `19079`
- `mean_candidate_count`: `64.0`
- `tie_event_count`: `19079`
- `tie_event_fraction`: `1.0`
- `median_optimal_set_size`: `64.0`
- `mean_optimal_set_fraction`: `0.9932132514806856`
- `distinct_target_value_count`: `{'mean': 1.3697782902667854, 'median': 1.0, 'p90': 2.0, 'min': 1.0, 'max': 2.0}`
- `strict_distinct_margin`: `{'mean': 1.0, 'median': 1.0, 'p90': 1.0, 'min': 1.0, 'max': 1.0}`
- `ordinary_margin_zero_fraction`: `1.0`
- `target_entropy_bits`: `{'mean': 0.04833896627328329, 'median': 0.0, 'p90': 0.11611507530476972, 'min': 0.0, 'max': 0.27297085791406983}`
- `target_spread`: `{'mean': 0.3697782902667855, 'median': 0.0, 'p90': 1.0, 'min': 0.0, 'max': 1.0}`

## Longer-Horizon Resolution

### H_long=8
- `h_long`: `8`
- `tie_event_count`: `19079`
- `fraction_ties_broken`: `0.14188374652759578`
- `longer_horizon_spread`: `{'mean': 0.14188374652759578, 'median': 0.0, 'p90': 1.0, 'min': 0.0, 'max': 1.0}`
- `deterministic_longer_horizon_best_fraction`: `0.9375229309712249`
- `deterministic_longer_horizon_regret`: `{'mean': 0.06247706902877509, 'median': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 1.0}`
- `learned_choice_in_h4_tie_fraction`: `0.9308663976099376`
- `learned_longer_horizon_best_fraction`: `0.9725225225225225`
- `learned_longer_horizon_regret`: `{'mean': 0.027477477477477478, 'median': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 1.0}`

### H_long=16
- `h_long`: `16`
- `tie_event_count`: `19079`
- `fraction_ties_broken`: `0.2761151003721369`
- `longer_horizon_spread`: `{'mean': 0.2761151003721369, 'median': 0.0, 'p90': 1.0, 'min': 0.0, 'max': 1.0}`
- `deterministic_longer_horizon_best_fraction`: `0.8748362073483935`
- `deterministic_longer_horizon_regret`: `{'mean': 0.1251637926516065, 'median': 0.0, 'p90': 1.0, 'min': 0.0, 'max': 1.0}`
- `learned_choice_in_h4_tie_fraction`: `0.9308663976099376`
- `learned_longer_horizon_best_fraction`: `0.9460022522522522`
- `learned_longer_horizon_regret`: `{'mean': 0.053997747747747744, 'median': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 1.0}`

### H_long=32
- `h_long`: `32`
- `tie_event_count`: `19079`
- `fraction_ties_broken`: `0.39588028722679386`
- `longer_horizon_spread`: `{'mean': 0.39588028722679386, 'median': 0.0, 'p90': 1.0, 'min': 0.0, 'max': 1.0}`
- `deterministic_longer_horizon_best_fraction`: `0.8139315477750406`
- `deterministic_longer_horizon_regret`: `{'mean': 0.18606845222495938, 'median': 0.0, 'p90': 1.0, 'min': 0.0, 'max': 1.0}`
- `learned_choice_in_h4_tie_fraction`: `0.9308663976099376`
- `learned_longer_horizon_best_fraction`: `0.9157657657657657`
- `learned_longer_horizon_regret`: `{'mean': 0.08423423423423423, 'median': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 1.0}`
