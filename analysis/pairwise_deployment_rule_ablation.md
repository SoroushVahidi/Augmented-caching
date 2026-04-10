# Pairwise deployment-rule ablation

## Setup
- Pairwise scorer fixed to `evict_value_pairwise_v1` artifact trained on train split candidate-pair labels.
- Conditions: unguarded pairwise, pairwise+LRU black-box combiner, pairwise+LRU shortlist pre-filter (m sweep), pairwise+guard wrapper fallback.

## Overall online mean misses (lower is better)
- pairwise_lru_blackbox_combiner: 3.0000
- evict_value_pairwise_v1: 3.3333
- evict_value_v1: 3.3333
- lru: 3.3333
- pairwise_guard_wrapper_lru: 3.3333
- pairwise_shortlist_lru_m2: 3.3333
- pairwise_shortlist_lru_m4: 3.3333
- pairwise_shortlist_lru_m8: 3.3333
- pairwise_shortlist_lru_mfull: 3.3333
- pairwise_unguarded: 3.3333
- predictive_marker: 3.3333
- rest_v1: 3.3333
- trust_and_doubt: 3.3333

## Required direct answers
1. Unguarded pairwise collapse vs guarded? no (unguarded=3.3333, guarded=3.3333).
2. Simple combiner recovers most robust-baseline gap? yes (combiner=3.0000, robust_best=3.3333).
3. Heuristic shortlist size materially affects performance? no (best=pairwise_shortlist_lru_m2:3.3333).
4. Remaining weakness likely deployment-rule or scorer quality? deployment-rule dominated.
5. Strongest next default deployment rule: pairwise_lru_blackbox_combiner.

## Diagnostics
- Pairwise disagreement rate vs LRU: 1.0000
- Pairwise disagreement rate vs predictive_marker: 1.0000
- Pairwise inversion-style mistake rate: 0.0000 (0/4 pair comparisons).
- Inversion definition: For each labeled pair (i,j), count a mistake when predicted preference p(i better than j)>=0.5 disagrees with rollout label_i_better.

## Per-family mean misses
- general: pairwise_lru_blackbox_combiner=3.000, evict_value_pairwise_v1=3.333, evict_value_v1=3.333, lru=3.333, pairwise_guard_wrapper_lru=3.333, pairwise_shortlist_lru_m2=3.333, pairwise_shortlist_lru_m4=3.333, pairwise_shortlist_lru_m8=3.333, pairwise_shortlist_lru_mfull=3.333, pairwise_unguarded=3.333, predictive_marker=3.333, rest_v1=3.333, trust_and_doubt=3.333

## Per-capacity mean misses
- cap=2: pairwise_lru_blackbox_combiner=3.000, evict_value_pairwise_v1=4.000, evict_value_v1=4.000, lru=4.000, pairwise_guard_wrapper_lru=4.000, pairwise_shortlist_lru_m2=4.000, pairwise_shortlist_lru_m4=4.000, pairwise_shortlist_lru_m8=4.000, pairwise_shortlist_lru_mfull=4.000, pairwise_unguarded=4.000, predictive_marker=4.000, rest_v1=4.000, trust_and_doubt=4.000
- cap=3: evict_value_pairwise_v1=3.000, evict_value_v1=3.000, lru=3.000, pairwise_guard_wrapper_lru=3.000, pairwise_lru_blackbox_combiner=3.000, pairwise_shortlist_lru_m2=3.000, pairwise_shortlist_lru_m4=3.000, pairwise_shortlist_lru_m8=3.000, pairwise_shortlist_lru_mfull=3.000, pairwise_unguarded=3.000, predictive_marker=3.000, rest_v1=3.000, trust_and_doubt=3.000
- cap=4: evict_value_pairwise_v1=3.000, evict_value_v1=3.000, lru=3.000, pairwise_guard_wrapper_lru=3.000, pairwise_lru_blackbox_combiner=3.000, pairwise_shortlist_lru_m2=3.000, pairwise_shortlist_lru_m4=3.000, pairwise_shortlist_lru_m8=3.000, pairwise_shortlist_lru_mfull=3.000, pairwise_unguarded=3.000, predictive_marker=3.000, rest_v1=3.000, trust_and_doubt=3.000
