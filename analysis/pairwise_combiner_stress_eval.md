# Pairwise combiner stress evaluation

## Stress setup
- Traces used: 4 (example.json, example_atlas_v1.json, example_general_caching.json, example_unweighted.json).
- Split counts (trace-level): {'train': 2, 'val': 1, 'test': 1}.
- Capacities: [2, 3, 4, 5, 6]; horizons: [4, 8, 12]; max_requests_per_trace=0 (0 means full).

## Overall online mean misses
- pairwise_lru_blackbox_combiner: 3.0000
- evict_value_pairwise_v1: 3.2000
- evict_value_v1: 3.2000
- lru: 3.2000
- pairwise_guard_wrapper_lru: 3.2000
- pairwise_shortlist_lru_m4: 3.2000
- pairwise_shortlist_lru_mfull: 3.2000
- pairwise_unguarded: 3.2000
- predictive_marker: 3.2000
- rest_v1: 3.2000
- trust_and_doubt: 3.2000

## Required direct answers
1. Combiner still beats pairwise_unguarded? yes (combiner=3.0000, unguarded=3.2000).
2. Combiner beats/ties strong robust baselines? yes (combiner=3.0000, best_robust=3.2000).
3. Gain broad across families or concentrated? broad (1/1 families where combiner < unguarded).
4. Combiner help mode: catastrophic-correction dominated (choose_lru_fraction=0.0286).
5. Strongest next bottleneck: data scale.
