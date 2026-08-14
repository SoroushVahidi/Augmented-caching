# Controlled Timing — Interpretation Caveat (2026-08-13)

## Two separate timing evidence sources — do not merge them

**1. The controlled 420-row, 5-repetition campaign** (`raw_timing_runs.csv`, job
`1171758`) covers exactly four policies, each with 105 measurements (7 families × 3
capacities × 5 repetitions):

- `lru`
- `fifo_reinsertion`
- `sieve`
- `halp_causal`

This is the only part of the timing evidence with repetition variance (stddev, 95% CI)
and cross-policy statistical comparison. See `controlled_timing_summary.csv` and
`controlled_timing_integrity.md`.

**2. `evict_value_v1`'s runtime** is a **single, non-repeated** wall-clock measurement
recorded during the corrected held-out treatment run itself
(`evict_value_v1_final_42_20260810/policy_comparison.csv`, `runtime_seconds` column),
not a product of the controlled timing campaign. Example: brightkite/cap32,
`primary_controlled_window` → 1411.5893 s / 40000 scored requests = 35289.73
µs/request.

## Why `evict_value_v1` was not included in the 5-repetition campaign

Per `TIMING_PROTOCOL.md` (transferred, hash-verified) and `timing_summary.csv`'s own
note, a fresh controlled 5-repetition timing run of `evict_value_v1` was judged
infeasible: the single held-out treatment run already took on the order of
tens of minutes per family/capacity cell (`runtime_seconds` up to ~1412s in the example
above), so 5 repetitions × 7 families × 3 capacities would have been a very high
additional compute cost for a policy whose primary scientific evidence
(miss ratio, regret vs. baselines) was already fully collected in the held-out
treatment campaign.

## Claim-safety rule

**Never** place `evict_value_v1` as a fifth row in the 420-row / 4-policy controlled
timing table, and never compute a "slowdown vs. LRU" or similar ratio for
`evict_value_v1` using the controlled campaign's LRU mean — the two measurements come
from different runs, different scoring windows in one case, and critically
`evict_value_v1`'s number has **no repetition variance**, so no confidence interval or
statistical comparison against the four controlled policies is defensible.

**Allowed claim:** "The four baseline/simple policies (LRU, FIFO-Reinsertion, SIEVE,
HALP-causal) were timed under a controlled 5-repetition, 420-measurement campaign.
`evict_value_v1`'s runtime is reported separately, as a single measurement taken during
its own held-out evaluation run, several orders of magnitude slower than LRU/FIFO/SIEVE
and roughly comparable in order of magnitude to (but not a repeated, directly
statistically comparable to) HALP-causal's controlled mean."

**Prohibited claim:** "`evict_value_v1` was timed under the same controlled
5-repetition protocol as LRU/FIFO/SIEVE/HALP-causal" — it was not.
