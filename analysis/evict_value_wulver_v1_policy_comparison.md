# evict_value_v1 Wulver policy comparison

## Aggregate mean misses (all traces × capacities in run)
- **atlas_v3:** 30951.9524
- **blind_oracle:** 40352.2857
- **blind_oracle_lru_combiner:** 30953.2381
- **evict_value_v1:** 32166.6667
- **lru:** 30951.9524
- **ml_gate_v1:** 30951.9524
- **ml_gate_v2:** 30951.9524
- **predictive_marker:** 30801.1905
- **rest_v1:** 30951.9524
- **trust_and_doubt:** 30986.0000

## Relative vs LRU (lower misses is better; positive % = fewer misses than LRU)
- atlas_v3: 0.00% vs LRU
- blind_oracle: -30.37% vs LRU
- blind_oracle_lru_combiner: -0.00% vs LRU
- evict_value_v1: -3.92% vs LRU
- lru: 0.00% vs LRU
- ml_gate_v1: 0.00% vs LRU
- ml_gate_v2: 0.00% vs LRU
- predictive_marker: 0.49% vs LRU
- rest_v1: 0.00% vs LRU
- trust_and_doubt: -0.11% vs LRU

## Relative vs rest_v1
- atlas_v3: 0.00% vs rest_v1
- blind_oracle: -30.37% vs rest_v1
- blind_oracle_lru_combiner: -0.00% vs rest_v1
- evict_value_v1: -3.92% vs rest_v1
- lru: 0.00% vs rest_v1
- ml_gate_v1: 0.00% vs rest_v1
- ml_gate_v2: 0.00% vs rest_v1
- predictive_marker: 0.49% vs rest_v1
- rest_v1: 0.00% vs rest_v1
- trust_and_doubt: -0.11% vs rest_v1

## Per-family mean misses (evict_value_v1 vs LRU vs rest_v1)
- **brightkite:** evict_value_v1=18153.67, lru=16700.00, rest_v1=16700.00 (loss vs best baseline here)
- **citibike:** evict_value_v1=19353.00, lru=18694.00, rest_v1=18694.00 (loss vs best baseline here)
- **cloudphysics:** evict_value_v1=48967.67, lru=48532.00, rest_v1=48532.00 (loss vs best baseline here)
- **metacdn:** evict_value_v1=29269.00, lru=28602.33, rest_v1=28602.33 (loss vs best baseline here)
- **metakv:** evict_value_v1=38038.67, lru=38029.67, rest_v1=38029.67 (loss vs best baseline here)
- **twemcache:** evict_value_v1=21384.67, lru=16105.67, rest_v1=16105.67 (loss vs best baseline here)
- **wiki2018:** evict_value_v1=50000.00, lru=50000.00, rest_v1=50000.00 (tie vs best baseline here)

- evict_value_v1 model: `models/evict_value_wulver_v1_best.pkl`
