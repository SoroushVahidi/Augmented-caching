# evict_value_v1 failure-slice audit

## Run scope
- Traces processed: 2
- Capacities: 2, 3, 4
- Eviction-decision rows (anchored on evict_value_v1): 27
- Skipped policies: none

## Overall comparison counts vs each competitor
- predictive_marker: disagree=25/27 (92.6%), evict_value_v1 loses=7, wins=0, tie=20, evict_hits=0, predictive_marker_hits=7
- trust_and_doubt: disagree=25/27 (92.6%), evict_value_v1 loses=8, wins=0, tie=19, evict_hits=0, trust_and_doubt_hits=8
- rest_v1: disagree=14/27 (51.9%), evict_value_v1 loses=6, wins=0, tie=21, evict_hits=0, rest_v1_hits=6
- lru: disagree=14/27 (51.9%), evict_value_v1 loses=2, wins=0, tie=25, evict_hits=0, lru_hits=2

## Per-family breakdown
- legacy (rows=27): predictive_marker:7, trust_and_doubt:8, rest_v1:6, lru:2

## Per-capacity breakdown
- cap=2 (rows=16): predictive_marker:3, trust_and_doubt:3, rest_v1:2, lru:0
- cap=3 (rows=8): predictive_marker:2, trust_and_doubt:3, rest_v1:2, lru:0
- cap=4 (rows=3): predictive_marker:2, trust_and_doubt:2, rest_v1:2, lru:2

## Disagreement-slice breakdown
- disagree_with_0_competitors: 1 (3.7%)
- disagree_with_1_competitors: 1 (3.7%)
- disagree_with_2_competitors: 10 (37.0%)
- disagree_with_3_competitors: 3 (11.1%)
- disagree_with_4_competitors: 12 (44.4%)

## Low-margin vs high-margin breakdown
- Score margin was not available in diagnostics for this run; all rows have null margin.

## Top recurring failure patterns
- loses_to=predictive_marker+trust_and_doubt|pos=mid|req_bucket=None: 1
- loses_to=trust_and_doubt|pos=late|req_bucket=None: 1
- loses_to=predictive_marker|pos=late|req_bucket=None: 1
- loses_to=predictive_marker+trust_and_doubt|pos=late|req_bucket=None: 1
- loses_to=predictive_marker+rest_v1+trust_and_doubt|pos=mid|req_bucket=0: 1
- loses_to=rest_v1|pos=late|req_bucket=1: 1
- loses_to=rest_v1+trust_and_doubt|pos=late|req_bucket=1: 1
- loses_to=predictive_marker+rest_v1+trust_and_doubt|pos=late|req_bucket=3: 1
- loses_to=lru+predictive_marker+rest_v1+trust_and_doubt|pos=late|req_bucket=1: 1
- loses_to=lru+predictive_marker+rest_v1+trust_and_doubt|pos=late|req_bucket=3: 1

## Notes
- Audit rows are anchored on steps where evict_value_v1 performs an eviction decision (cache full + miss).
- Candidate spread summaries are populated only when diagnostics expose them; otherwise null by design.
