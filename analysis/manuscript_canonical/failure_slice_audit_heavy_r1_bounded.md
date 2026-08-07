# evict_value_v1 failure-slice audit

## Run scope
- Traces processed: 3
- Capacities: 32
- Eviction-decision rows (anchored on evict_value_v1): 0
- Skipped policies:
  - evict_value_v1: UnpicklingError: invalid load key, '\x0d'.

## Overall comparison counts vs each competitor
- predictive_marker: disagree=0/0 (0.0%), evict_value_v1 loses=0, wins=0, tie=0, evict_hits=0, predictive_marker_hits=0
- trust_and_doubt: disagree=0/0 (0.0%), evict_value_v1 loses=0, wins=0, tie=0, evict_hits=0, trust_and_doubt_hits=0
- rest_v1: disagree=0/0 (0.0%), evict_value_v1 loses=0, wins=0, tie=0, evict_hits=0, rest_v1_hits=0
- lru: disagree=0/0 (0.0%), evict_value_v1 loses=0, wins=0, tie=0, evict_hits=0, lru_hits=0

## Per-family breakdown

## Per-capacity breakdown

## Disagreement-slice breakdown

## Low-margin vs high-margin breakdown
- Score margin was not available in diagnostics for this run; all rows have null margin.

## Top recurring failure patterns
- No rows where a competitor hits and evict_value_v1 misses inside the audited eviction slices.

## Notes
- Audit rows are anchored on steps where evict_value_v1 performs an eviction decision (cache full + miss).
- Candidate spread summaries are populated only when diagnostics expose them; otherwise null by design.
