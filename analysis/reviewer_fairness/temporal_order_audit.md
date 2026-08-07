# Temporal Order Audit — `evict_value_v1_fair_v1` training corpus

**Finding: mixed, family-specific. fair_v1's implicit assumption ("later raw
row position = later chronological time") is confirmed TRUE for 3 of 7
families, UNVERIFIABLE for 2, and demonstrably FALSE for 2.** See
`analysis/reviewer_fairness/temporal_order_audit.json` for the full
per-family evidence; this is the narrative summary.

## Method

For every family, every record's timestamp (where one exists in the
canonical schema) was read directly — not sampled, not assumed — from
both the canonical `[0,50000)` trace and the fair_v1 `[50000,100000)`
training segment, and checked for strict non-decreasing order across the
**entire** segment (not just first/last row, which can hide non-monotonic
order in between). An initial pass had a string-vs-numeric comparison bug
for twemcache's small-integer timestamp field (`"99" < "187"`
lexicographically, which is wrong); this was caught and fixed before
drawing conclusions.

## Results by family

| Family | Raw order chronological? | Relationship to test | Confidence |
|---|---|---|---|
| twemcache | **Yes** (verified) | LATER — contiguous, confirmed future data | High |
| metacdn | **Yes** (verified) | LATER — contiguous, confirmed future data | High |
| cloudphysics | **Yes** (verified) | LATER — contiguous, confirmed future data | High |
| brightkite | **No** | Interleaved/unknown — near-total calendar overlap | High |
| citibike | **No** | Interleaved/unknown — near-total calendar overlap | High |
| wiki2018 | N/A — no time axis | Not applicable | High (evidence of absence) |
| metakv | Unverifiable | Unknown | Low |

## Family-specific evidence

- **brightkite**: the raw file (`loc-brightkite_totalCheckins.txt.gz`) is
  grouped by `user_id` — the first 5 raw lines are all user 0's checkins,
  listed in *reverse* chronological order within that user, before user 1
  begins. Row position has no relationship to time. Directly confirmed:
  the canonical `[0,50000)` segment's own timestamps are non-monotonic
  (span 2008-03-22 to 2010-10-17, non-decreasing check fails), and the
  fair_v1 training segment's calendar range (2008-04-14 to 2010-10-17)
  almost entirely overlaps the canonical range.
- **citibike**: `202401-citibike-tripdata_1.csv` (the file this pipeline's
  parser selects) is a large multi-station export whose row order is not
  monotonic in `started_at` across the checked range; both segments span
  essentially the same Jan 1–14, 2024 window.
- **wiki2018**: the raw file's `timestamp` column is empty for every row
  and the file is sorted **alphabetically by article name** — confirmed by
  direct inspection (`en:!!!`, `en:!!!_(album)`, ...). This is a static
  per-article catalog, not a timestamped request log; "earlier" and
  "later" are not meaningful descriptions of its row order at all.
- **metakv**: the raw CSV header is `key,op,size,op_count,key_size` — no
  timestamp field exists in the source. Cannot be verified either way.
- **twemcache / metacdn / cloudphysics**: all three have numeric
  timestamps that are strictly non-decreasing across the full checked
  range, and in every case the canonical segment's last timestamp and the
  training segment's first timestamp are immediately adjacent (2ms apart
  for metacdn, 31µs for cloudphysics, contiguous integer boundary for
  twemcache) — genuinely, cleanly confirmed as **future data relative to
  the test period**.

## Consequence

`evict_value_v1_fair_v1`'s training corpus **cannot** be certified as a
uniform, defensible "later same-family data" (Option D) protocol across
all 7 families, because that framing is only actually true for 3 of them.
For brightkite and citibike, the correct classification is closer to
"unverified, likely-interleaved order" — which is a *different* problem
from "future data" and requires a different fix (re-sorting by time before
splitting, not simply picking a later row range). For wiki2018 and metakv,
neither "future" nor "past" is a meaningful description of the fair_v1
split at all.

See `docs/reviewer_fairness_temporal_rules.md` for the frozen eligibility
rules this audit is checked against, and the main final report for the
protocol decision.
