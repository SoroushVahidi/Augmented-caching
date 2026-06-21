# evict_value_v1 Wulver policy comparison

## Aggregate mean misses (all traces × capacities in run)
- **blind_oracle_lru_combiner:** 32496.8571
- **evict_value_v1:** 36316.5714
- **fifo_reinsertion:** 32530.5714
- **lru:** 32495.5714
- **predictive_marker:** 32664.5714
- **rest_v1:** 32495.5714
- **sieve:** 33279.2857
- **trust_and_doubt:** 32928.8571

## Relative vs LRU (lower misses is better; positive % = fewer misses than LRU)
- blind_oracle_lru_combiner: -0.00% vs LRU
- evict_value_v1: -11.76% vs LRU
- fifo_reinsertion: -0.11% vs LRU
- lru: 0.00% vs LRU
- predictive_marker: -0.52% vs LRU
- rest_v1: 0.00% vs LRU
- sieve: -2.41% vs LRU
- trust_and_doubt: -1.33% vs LRU

## Relative vs rest_v1
- blind_oracle_lru_combiner: -0.00% vs rest_v1
- evict_value_v1: -11.76% vs rest_v1
- fifo_reinsertion: -0.11% vs rest_v1
- lru: 0.00% vs rest_v1
- predictive_marker: -0.52% vs rest_v1
- rest_v1: 0.00% vs rest_v1
- sieve: -2.41% vs rest_v1
- trust_and_doubt: -1.33% vs rest_v1

## Per-family mean misses (evict_value_v1 vs LRU vs rest_v1)
- **brightkite:** evict_value_v1=23360.00, lru=15479.00, rest_v1=15479.00 (loss vs best baseline here)
- **citibike:** evict_value_v1=24914.00, lru=17124.00, rest_v1=17124.00 (loss vs best baseline here)
- **cloudphysics:** evict_value_v1=49230.00, lru=47748.00, rest_v1=47748.00 (loss vs best baseline here)
- **metacdn:** evict_value_v1=32711.00, lru=27811.00, rest_v1=27811.00 (loss vs best baseline here)
- **metakv:** evict_value_v1=38273.00, lru=37984.00, rest_v1=37984.00 (loss vs best baseline here)
- **twemcache:** evict_value_v1=35728.00, lru=31323.00, rest_v1=31323.00 (loss vs best baseline here)
- **wiki2018:** evict_value_v1=50000.00, lru=50000.00, rest_v1=50000.00 (tie vs best baseline here)

- evict_value_v1 model: `models/evict_value_wulver_v1_best_heavy_r1.pkl`
