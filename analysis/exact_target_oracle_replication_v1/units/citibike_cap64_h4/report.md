# Exact-Target-Oracle Diagnostic

Status: `COMPLETE`

This diagnostic compares LRU, the exact finite-horizon eviction-loss target oracle, the eligible learned eviction-loss scalar policy when available, and offline Belady as a separate future-aware reference.

## Scores

| policy | information class | misses | miss ratio | excess vs LRU | gap to Belady |
|---|---:|---:|---:|---:|---:|
| lru | ONLINE_DEPLOYABLE | 15709 | 0.392725 | 0 | 3785 |
| exact_finite_horizon_eviction_loss_oracle | FUTURE_AWARE_DIAGNOSTIC_TARGET_ORACLE | 24678 | 0.616950 | 8969 | 12754 |
| offline_belady | FUTURE_AWARE_REFERENCE | 11924 | 0.298100 | -3785 | 0 |

## Learned-Model Eligibility

Status: `NOT_AVAILABLE`

Reason: `disabled by replication protocol; optional learned comparison not required`
