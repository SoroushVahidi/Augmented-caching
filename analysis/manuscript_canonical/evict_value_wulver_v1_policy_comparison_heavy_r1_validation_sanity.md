# evict_value_v1 Wulver policy comparison

## Aggregate mean misses (all traces × capacities in run)
- **evict_value_v1:** 1837.0000
- **lru:** 1643.0000

## Relative vs LRU (lower misses is better; positive % = fewer misses than LRU)
- evict_value_v1: -11.81% vs LRU
- lru: 0.00% vs LRU

## Relative vs rest_v1
- evict_value_v1: 0.00% vs rest_v1
- lru: 0.00% vs rest_v1

## Per-family mean misses (evict_value_v1 vs LRU vs rest_v1)
- **brightkite:** evict_value_v1=1837.00, lru=1643.00, rest_v1=0.00 (loss vs best baseline here)

- evict_value_v1 model: `models/evict_value_wulver_v1_best_heavy_r1.pkl`
