# ml_gate_v1 first check

Experimental, proof-of-concept, lightweight learned gate.

## Training data
- Traces: data/example_unweighted.json + data/example_atlas_v1.json (sample-first mode by default).
- Capacities: 2/3/4.
- Label horizon H=4.

## Label definition
- y=1 iff predictor-victim penalty < LRU-victim penalty, where penalty=1 if the evicted page returns within H requests, else 0.
- ties map to y=0 (abstain/fallback preference).

## Feature set
- request bucket/confidence
- predictor and LRU scores
- predictor-LRU disagreement and score gap
- predictor/LRU recency ranks
- bucket and confidence gaps
- cache bucket diversity/mean/std
- recent regret/disagreement rates

## Model
- LogisticRegression(class_weight='balanced') + StandardScaler.
- Metrics JSON: analysis/ml_gate_v1_metrics.json.

## Train/val/test metrics
- train: accuracy=1.000 precision=1.000 recall=1.000 f1=1.000 auc=1.000 class_balance=0.231
- val: accuracy=0.600 precision=0.000 recall=0.000 f1=0.000 auc=0.500 class_balance=0.000
- test: accuracy=0.500 precision=0.000 recall=0.000 f1=0.000 auc=0.500 class_balance=0.000

## Baseline comparison (mean misses over first-check traces)
- ml_gate_v1: 6.917
- rest_v1: 6.583
- atlas_v3: 7.000
- lru: 6.917
- blind_oracle: 7.167
- predictive_marker: 6.167
- trust_and_doubt: 6.000
- blind_oracle_lru_combiner: 7.250

## Honesty check
- This is a small-sample POC; if gains are small or negative, treat this as plumbing validation rather than final evidence.
- Full-scale trace-family training/evaluation should be run on Wulver.
