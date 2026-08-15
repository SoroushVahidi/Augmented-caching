# Exact-Target-Oracle Diagnostic

Status: `COMPLETE`

This diagnostic compares LRU, the exact finite-horizon eviction-loss target oracle, the eligible learned eviction-loss scalar policy when available, and offline Belady as a separate future-aware reference.

## Scores

| policy | information class | misses | miss ratio | excess vs LRU | gap to Belady |
|---|---:|---:|---:|---:|---:|
| lru | ONLINE_DEPLOYABLE | 14426 | 0.360650 | 0 | 2520 |
| exact_finite_horizon_eviction_loss_oracle | FUTURE_AWARE_DIAGNOSTIC_TARGET_ORACLE | 19153 | 0.478825 | 4727 | 7247 |
| offline_belady | FUTURE_AWARE_REFERENCE | 11906 | 0.297650 | -2520 | 0 |

## Learned-Model Eligibility

Status: `NOT_AVAILABLE`

Reason: `disabled by replication protocol; optional learned comparison not required`
