# evict_value_v1 Wulver policy comparison

## Aggregate mean misses (all traces × capacities in run)
- **blind_oracle:** 29554.0000
- **blind_oracle_lru_combiner:** 14759.0000
- **evict_value_v1:** 20533.0000
- **lru:** 14758.0000
- **predictive_marker:** 14923.0000
- **rest_v1:** 14758.0000
- **trust_and_doubt:** 15306.0000

## Relative vs LRU (lower misses is better; positive % = fewer misses than LRU)
- blind_oracle: -100.26% vs LRU
- blind_oracle_lru_combiner: -0.01% vs LRU
- evict_value_v1: -39.13% vs LRU
- lru: 0.00% vs LRU
- predictive_marker: -1.12% vs LRU
- rest_v1: 0.00% vs LRU
- trust_and_doubt: -3.71% vs LRU

## Relative vs rest_v1
- blind_oracle: -100.26% vs rest_v1
- blind_oracle_lru_combiner: -0.01% vs rest_v1
- evict_value_v1: -39.13% vs rest_v1
- lru: 0.00% vs rest_v1
- predictive_marker: -1.12% vs rest_v1
- rest_v1: 0.00% vs rest_v1
- trust_and_doubt: -3.71% vs rest_v1

## Per-family mean misses (evict_value_v1 vs LRU vs rest_v1)
- **brightkite:** evict_value_v1=20533.00, lru=14758.00, rest_v1=14758.00 (loss vs best baseline here)

- evict_value_v1 model: `models/evict_value_wulver_v1_best_heavy_r1.pkl`
