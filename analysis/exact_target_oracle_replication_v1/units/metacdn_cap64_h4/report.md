# Exact-Target-Oracle Diagnostic

Status: `COMPLETE`

This diagnostic compares LRU, the exact finite-horizon eviction-loss target oracle, the eligible learned eviction-loss scalar policy when available, and offline Belady as a separate future-aware reference.

## Scores

| policy | information class | misses | miss ratio | excess vs LRU | gap to Belady |
|---|---:|---:|---:|---:|---:|
| lru | ONLINE_DEPLOYABLE | 22586 | 0.564650 | 0 | 2577 |
| exact_finite_horizon_eviction_loss_oracle | FUTURE_AWARE_DIAGNOSTIC_TARGET_ORACLE | 26029 | 0.650725 | 3443 | 6020 |
| offline_belady | FUTURE_AWARE_REFERENCE | 20009 | 0.500225 | -2577 | 0 |

## Learned-Model Eligibility

Status: `NOT_AVAILABLE`

Reason: `disabled by replication protocol; optional learned comparison not required`
