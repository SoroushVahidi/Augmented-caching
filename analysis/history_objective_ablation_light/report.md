# History-feature objective ablation (lightweight)

## Setup
- Reuses the existing history-aware experimental path and compact `repo_light` trace subset.
- Base and history-aware feature pipelines are fixed.
- Compared objectives on history-aware arm: replay-loss regression vs pairwise decision objective.
- Model family kept aligned with histogram gradient boosting (regressor/classifier variants).

## Candidate-ranking metrics (test)
- base_regression: top1=0.5000, mean_regret=0.0000
- history_regression: top1=0.3750, mean_regret=0.1250
- history_pairwise: top1=0.3750, mean_regret=0.0000

## Downstream replay misses (mean)
- base_regression: 5.556
- history_regression: 5.389
- history_pairwise: 5.333

## Answer on prior mixed history-feature result
- Evidence in this lightweight run suggests the earlier mixed result was more due to objective/loss mismatch than features alone.
- This remains a lightweight result and should not be interpreted as heavy_r1 manuscript evidence.
