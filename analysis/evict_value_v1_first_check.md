# evict_value_v1 first check

## Core target and policy
- Candidate-centric regression target: y_loss(q,t;H) = misses over next H requests after forcing eviction of candidate q and replaying LRU transitions.
- Online policy evicts candidate with minimum predicted loss.

## Model sweep summary
- Winner by validation mean regret: ridge
- ridge: val_mae=1.669, test_mae=1.211, val_top1=0.364, test_top1=0.412
- random_forest: val_mae=1.626, test_mae=1.256, val_top1=0.318, test_top1=0.294
- hist_gb: val_mae=1.828, test_mae=1.471, val_top1=0.318, test_top1=0.353

## Policy comparison (mean misses)
- evict_value_v1: 5.867
- rest_v1: 6.467
- ml_gate_v2: 6.933
- ml_gate_v1: 6.800
- atlas_v3: 6.867
- lru: 6.933
- blind_oracle: 7.000
- predictive_marker: 6.267
- trust_and_doubt: 6.000
- blind_oracle_lru_combiner: 7.133

## Answers
1. Direct candidate scoring better than trust-gating? yes
2. Candidate-centric supervision better than gate refinements? yes
3. Which horizon works best? this run trained horizon=8; run the training sweep across horizons to confirm global best.
4. Which model family works best? ridge
5. Does evict_value_v1 beat rest_v1? yes
6. If not, clearest bottleneck: small data/trace diversity and limited features around long-range reuse interactions.

## Honesty
- This is a first proof-of-concept. If rest_v1 still wins, direct candidate prediction is not yet sufficient at this data scale.
- Environment is sufficient for smoke/first-check; broader claims should be re-run on larger Wulver-scale traces.
