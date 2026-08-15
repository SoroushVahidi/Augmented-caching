# Exact-Target-Oracle Diagnostic

Status: `COMPLETE`

This diagnostic compares LRU, the exact finite-horizon eviction-loss target oracle, the eligible learned eviction-loss scalar policy when available, and offline Belady as a separate future-aware reference.

## Scores

| policy | information class | misses | miss ratio | excess vs LRU | gap to Belady |
|---|---:|---:|---:|---:|---:|
| lru | ONLINE_DEPLOYABLE | 12397 | 0.309925 | 0 | 1514 |
| exact_finite_horizon_eviction_loss_oracle | FUTURE_AWARE_DIAGNOSTIC_TARGET_ORACLE | 18971 | 0.474275 | 6574 | 8088 |
| offline_belady | FUTURE_AWARE_REFERENCE | 10883 | 0.272075 | -1514 | 0 |

## Learned-Model Eligibility

Status: `NOT_AVAILABLE`

Reason: `disabled by replication protocol; optional learned comparison not required`
