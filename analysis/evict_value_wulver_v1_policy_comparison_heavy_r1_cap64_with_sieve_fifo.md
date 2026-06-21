# evict_value_v1 Wulver policy comparison

## Aggregate mean misses (all traces × capacities in run)
- **blind_oracle_lru_combiner:** 33651.2857
- **evict_value_v1:** 34409.0000
- **fifo_reinsertion:** 33672.1429
- **lru:** 33650.1429
- **predictive_marker:** 33773.1429
- **rest_v1:** 33650.1429
- **sieve:** 34362.7143
- **trust_and_doubt:** 33998.7143

## Relative vs LRU (lower misses is better; positive % = fewer misses than LRU)
- blind_oracle_lru_combiner: -0.00% vs LRU
- evict_value_v1: -2.26% vs LRU
- fifo_reinsertion: -0.07% vs LRU
- lru: 0.00% vs LRU
- predictive_marker: -0.37% vs LRU
- rest_v1: 0.00% vs LRU
- sieve: -2.12% vs LRU
- trust_and_doubt: -1.04% vs LRU

## Relative vs rest_v1
- blind_oracle_lru_combiner: -0.00% vs rest_v1
- evict_value_v1: -2.26% vs rest_v1
- fifo_reinsertion: -0.07% vs rest_v1
- lru: 0.00% vs rest_v1
- predictive_marker: -0.37% vs rest_v1
- rest_v1: 0.00% vs rest_v1
- sieve: -2.12% vs rest_v1
- trust_and_doubt: -1.04% vs rest_v1

## Per-family mean misses (evict_value_v1 vs LRU vs rest_v1)
- **brightkite:** evict_value_v1=18316.00, lru=16543.00, rest_v1=16543.00 (loss vs best baseline here)
- **citibike:** evict_value_v1=19855.00, lru=18964.00, rest_v1=18964.00 (loss vs best baseline here)
- **cloudphysics:** evict_value_v1=49339.00, lru=48575.00, rest_v1=48575.00 (loss vs best baseline here)
- **metacdn:** evict_value_v1=29397.00, lru=28616.00, rest_v1=28616.00 (loss vs best baseline here)
- **metakv:** evict_value_v1=38044.00, lru=38041.00, rest_v1=38041.00 (loss vs best baseline here)
- **twemcache:** evict_value_v1=35912.00, lru=34812.00, rest_v1=34812.00 (loss vs best baseline here)
- **wiki2018:** evict_value_v1=50000.00, lru=50000.00, rest_v1=50000.00 (tie vs best baseline here)

- evict_value_v1 model: `models/evict_value_wulver_v1_best_heavy_r1.pkl`
