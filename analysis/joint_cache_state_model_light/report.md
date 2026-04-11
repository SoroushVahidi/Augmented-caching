# Lightweight joint cache-state model follow-up

- Data source: existing joint decision-state dataset (no format changes).
- Models compared: evict_value_v1 lightweight baseline, history-aware pairwise-light, joint softmax victim model.
- Replay mean misses: {'evict_value_v1_lightweight': 6.111111111111111, 'history_pairwise_light': 5.222222222222222, 'joint_softmax': 5.222222222222222}
- Conservative assessment: **not_better_than_pairwise**.

## Interpretation
On this compact subset, joint softmax does not beat the simpler history-aware pairwise direction; medium-scale follow-up is not yet justified.
