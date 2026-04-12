# evict_value_v1 Wulver policy comparison

## Aggregate mean misses (all traces × capacities in run)
- **blind_oracle:** 3404.0000
- **blind_oracle_lru_combiner:** 1447.0000
- **evict_value_v1:** 1483.5000
- **lru:** 1445.5000
- **predictive_marker:** 1477.2500
- **rest_v1:** 1445.5000
- **trust_and_doubt:** 1539.5000

## Relative vs LRU (lower misses is better; positive % = fewer misses than LRU)
- blind_oracle: -135.49% vs LRU
- blind_oracle_lru_combiner: -0.10% vs LRU
- evict_value_v1: -2.63% vs LRU
- lru: 0.00% vs LRU
- predictive_marker: -2.20% vs LRU
- rest_v1: 0.00% vs LRU
- trust_and_doubt: -6.50% vs LRU

## Relative vs rest_v1
- blind_oracle: -135.49% vs rest_v1
- blind_oracle_lru_combiner: -0.10% vs rest_v1
- evict_value_v1: -2.63% vs rest_v1
- lru: 0.00% vs rest_v1
- predictive_marker: -2.20% vs rest_v1
- rest_v1: 0.00% vs rest_v1
- trust_and_doubt: -6.50% vs rest_v1

## Per-family mean misses (evict_value_v1 vs LRU vs rest_v1)
- **brightkite:** evict_value_v1=1627.50, lru=1551.00, rest_v1=1551.00 (loss vs best baseline here)
- **citibike:** evict_value_v1=1339.50, lru=1340.00, rest_v1=1340.00 (win vs best baseline here)

- evict_value_v1 model: `models/evict_value_wulver_v1_best_heavy_smoke.pkl`
