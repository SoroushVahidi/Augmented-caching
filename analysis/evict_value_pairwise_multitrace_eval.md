# evict_value_pairwise multitrace eval

## A. Overall online mean misses
- evict_value_v1: 3.3333
- evict_value_pairwise_v1: 3.3333
- predictive_marker: 3.3333
- trust_and_doubt: 3.3333
- rest_v1: 3.3333
- lru: 3.3333

## B. Offline test metrics (pointwise vs pairwise)
- evict_value_v1: top1=1.0000, pairwise_accuracy=0.0000, mean_regret=0.0000
- evict_value_pairwise_v1: top1=1.0000, pairwise_accuracy=1.0000, mean_regret=0.0000

## C. Per-family breakdown
- general: evict_value_pairwise_v1=3.3333, evict_value_v1=3.3333, lru=3.3333, predictive_marker=3.3333, rest_v1=3.3333, trust_and_doubt=3.3333

## D. Per-capacity breakdown
- cap=2: evict_value_pairwise_v1=4.0000, evict_value_v1=4.0000, lru=4.0000, predictive_marker=4.0000, rest_v1=4.0000, trust_and_doubt=4.0000
- cap=3: evict_value_pairwise_v1=3.0000, evict_value_v1=3.0000, lru=3.0000, predictive_marker=3.0000, rest_v1=3.0000, trust_and_doubt=3.0000
- cap=4: evict_value_pairwise_v1=3.0000, evict_value_v1=3.0000, lru=3.0000, predictive_marker=3.0000, rest_v1=3.0000, trust_and_doubt=3.0000

## E. Per-horizon breakdown (offline top1)
- h=4: pointwise_top1=1.0000, pairwise_top1=1.0000
- h=8: pointwise_top1=1.0000, pairwise_top1=1.0000

## F. Hard-slice breakdown
- Hard-slice unavailable or no overlap with test split.

## G. Direct answers
- pairwise beats pointwise overall: no
- pairwise beats pointwise on hard slices: unclear
- pairwise beats rest_v1 online: no
- pairwise gap to predictive_marker (misses): 0.0000
- pairwise gap to trust_and_doubt (misses): 0.0000
- remaining weakness signal: if pairwise wins offline but not vs strongest baselines online, likely feature weakness and deployment-rule gap under current data scale.
