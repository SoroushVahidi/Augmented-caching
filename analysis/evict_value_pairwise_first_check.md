# evict_value pairwise first check

## Online policy comparison (mean misses)
- evict_value_v1: 7.083
- evict_value_pairwise_v1: 6.333
- predictive_marker: 6.167
- trust_and_doubt: 6.000
- rest_v1: 6.583
- lru: 6.917

## Offline decision quality
- evict_value_pairwise_v1: top1=1.0000, pairwise_acc=1.0000, mean_regret=0.0000

## Hard loss slice (evict_value_v1 losses in failure-slice audit)
- evict_value_pairwise_v1: top1=1.0000, pairwise_acc=1.0000, mean_regret=0.0000

## Bottleneck read
- If pairwise improves hard-slice top1/regret but not online misses, bottleneck likely feature coverage or horizon mismatch.
- If pairwise does not improve offline hard slices either, objective mismatch is likely not the only blocker.
