# evict_value_v1 Wulver policy comparison

## Aggregate mean misses (all traces × capacities in run)
- **blind_oracle_lru_combiner:** 34668.5714
- **evict_value_v1:** 36344.1429
- **fifo_reinsertion:** 34688.0000
- **lru:** 34667.1429
- **predictive_marker:** 34843.8571
- **rest_v1:** 34667.1429
- **sieve:** 35313.5714
- **trust_and_doubt:** 35087.0000

## Relative vs LRU (lower misses is better; positive % = fewer misses than LRU)
- blind_oracle_lru_combiner: -0.00% vs LRU
- evict_value_v1: -4.84% vs LRU
- fifo_reinsertion: -0.06% vs LRU
- lru: 0.00% vs LRU
- predictive_marker: -0.51% vs LRU
- rest_v1: 0.00% vs LRU
- sieve: -1.86% vs LRU
- trust_and_doubt: -1.21% vs LRU

## Relative vs rest_v1
- blind_oracle_lru_combiner: -0.00% vs rest_v1
- evict_value_v1: -4.84% vs rest_v1
- fifo_reinsertion: -0.06% vs rest_v1
- lru: 0.00% vs rest_v1
- predictive_marker: -0.51% vs rest_v1
- rest_v1: 0.00% vs rest_v1
- sieve: -1.86% vs rest_v1
- trust_and_doubt: -1.21% vs rest_v1

## Per-family mean misses (evict_value_v1 vs LRU vs rest_v1)
- **brightkite:** evict_value_v1=19970.00, lru=18078.00, rest_v1=18078.00 (loss vs best baseline here)
- **citibike:** evict_value_v1=21573.00, lru=19994.00, rest_v1=19994.00 (loss vs best baseline here)
- **cloudphysics:** evict_value_v1=49626.00, lru=49273.00, rest_v1=49273.00 (loss vs best baseline here)
- **metacdn:** evict_value_v1=35951.00, lru=29380.00, rest_v1=29380.00 (loss vs best baseline here)
- **metakv:** evict_value_v1=38710.00, lru=38064.00, rest_v1=38064.00 (loss vs best baseline here)
- **twemcache:** evict_value_v1=38579.00, lru=37881.00, rest_v1=37881.00 (loss vs best baseline here)
- **wiki2018:** evict_value_v1=50000.00, lru=50000.00, rest_v1=50000.00 (tie vs best baseline here)

- evict_value_v1 model: `models/evict_value_wulver_v1_best_heavy_r1.pkl`
