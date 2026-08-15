# Exact-Target-Oracle Diagnostic

Status: `COMPLETE`

This diagnostic compares LRU, the exact finite-horizon eviction-loss target oracle, the eligible learned eviction-loss scalar policy when available, and offline Belady as a separate future-aware reference.

## Scores

| policy | information class | misses | miss ratio | excess vs LRU | gap to Belady |
|---|---:|---:|---:|---:|---:|
| lru | ONLINE_DEPLOYABLE | 38237 | 0.955925 | 0 | 3449 |
| exact_finite_horizon_eviction_loss_oracle | FUTURE_AWARE_DIAGNOSTIC_TARGET_ORACLE | 39937 | 0.998425 | 1700 | 5149 |
| offline_belady | FUTURE_AWARE_REFERENCE | 34788 | 0.869700 | -3449 | 0 |

## Learned-Model Eligibility

Status: `NOT_AVAILABLE`

Reason: `disabled by replication protocol; optional learned comparison not required`
