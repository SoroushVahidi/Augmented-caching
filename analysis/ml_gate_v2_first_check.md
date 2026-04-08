# ml_gate_v2 first check

## Label definition
- Counterfactual local replay label: y_reg = loss_lru(H) - loss_pred(H), where each loss is miss count after forcing the eviction choice and replaying next H requests with LRU transitions.
- Binary target: y_cls = 1 if y_reg > margin else 0.

## Horizons and split
- Dataset built with horizons {4,8,16}; training sweep reported for selected horizon from metrics.json.
- Split method: hash(trace|capacity) into train/val/test buckets; fallback deterministic row split only if split buckets are empty.

## Model families
- logistic_regression: val_f1=0.667, test_f1=0.333, test_auc=0.346
- decision_tree: val_f1=0.000, test_f1=0.000, test_auc=0.500
- random_forest: val_f1=0.000, test_f1=0.000, test_auc=0.808
- hist_gb: val_f1=0.000, test_f1=0.000, test_auc=0.500
- Winner by val F1: logistic_regression

## Policy comparison (mean misses)
- ml_gate_v2: 6.933
- ml_gate_v1: 6.800
- rest_v1: 6.467
- atlas_v3: 6.867
- lru: 6.933
- predictive_marker: 6.267
- trust_and_doubt: 6.000

## Honesty
- If ml_gate_v2 does not beat rest_v1 consistently, bottleneck likely remains data scale/trace diversity and label noise under short local rollouts.
- This environment is adequate for first-check iteration; Wulver is justified for large-scale trace-family experiments.
