# Corrected Held-Out `evict_value_v1` Treatment — Local Sync Integrity (2026-08-13)

Compact integrity record for the Reviewer #2 Major 1 corrected leave-one-family-out
held-out treatment, synced from Wulver to the local workstation and independently
re-audited here (not just re-read from the Wulver-side audit doc).

## Provenance

See `heldout_treatment_provenance.md` for the full source path, hash manifest, and
transfer method. Summary: transferred by `rsync` from
`login02:/mmfs1/project/ikoutis/sv96/Augmented-caching/analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/`
to the local canonical path
`analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/`.
All 14 required files plus `model_registry.json` and
`EVICT_VALUE_HELDOUT_SYNTHESIS.md` matched their required SHA-256 exactly — **16/16
hash PASS**.

## Independent local re-audit (recomputed from raw CSV/JSON, not copied from Wulver doc)

| Gate | Result |
|---|---|
| Row count | 42/42 |
| Unique `(trace, capacity, policy_variant)` keys | 42/42, 0 duplicates |
| Families | 7/7 (brightkite, citibike, cloudphysics, metacdn, metakv, twemcache, wiki2018) |
| Capacities | 32/64/128, 14 rows each |
| `policy_variant` split | `primary_controlled_window` 21, `deployment_full_stream` 21 |
| `status` | 42/42 `ok` |
| NaN/Inf/malformed cells | 0 |
| `hits + misses == scored_requests` | 42/42 reconciled |
| `primary_controlled_window` score window | 42/42 rows `[10000, 50000)` (`score_start=10000`, `score_end=50000`) |
| `deployment_full_stream` score window | 42/42 rows `[0, 50000)` |
| Held-out family excluded from that fold's training set | 7/7 folds PASS |
| Held-out family ≠ validation family | 7/7 folds PASS |
| Validation family excluded from training set | 7/7 folds PASS |
| Training set size = 5, full 7-family coverage per fold | 7/7 folds PASS |
| `model_sha256` constant within held-out family across capacities/variants | 7/7 families PASS |
| `model_sha256` matches `model_registry.json` `model_artifact_sha256` per held-out family | 7/7 PASS, 0 mismatches |
| `trace_sha256` constant per trace name across all rows | PASS, 0 inconsistencies |
| `model_registry.json`: `MODEL_SELECTION_FROZEN` | `true`, `expected_model_count=7`, `actual_model_count=7`, `missing_folds=[]` |

**Local classification: `FINAL_VALIDATED_SYNCED`.**

## Protocol caveat — CORRECTED 2026-08-13 (later pass); see below

**Correction notice:** the paragraph originally here restated
`baseline_eligibility.csv`'s claim that "zero executed result artifacts
exist anywhere... under any protocol" for LRB/3L-Cache/HALP/CACHEUS, and
that LRU/SIEVE/FIFO "only exist" in a mismatched-window comparison. That
claim is **only true of Wulver's own filesystem** (which is all
`baseline_eligibility.csv`, itself a Wulver-produced artifact, ever
searched). It is **not true of this workstation**: `analysis/reviewer_fairness/`
(produced separately by `scripts/experiments/run_reviewer_fairness.py`, a
pipeline that has never run on Wulver) already contains exact-evaluation-
protocol results for all seven baselines — LRB, 3L-Cache, CACHEUS, HALP,
LRU, SIEVE, and FIFO-Reinsertion — each with 21/21 `primary_controlled_window`
cells whose `(trace_sha256, capacity)` key set is identical to this
treatment's 21 cells, same windows, same capacity/object-size semantics,
same `hits+misses=scored_requests` accounting, and `future_information=none`
on every row. See `major1_protocol_comparability.md` and
`major1_reviewer_summary.md` in this directory for the full audit and the
validated head-to-head comparison.

What remains true: **no baseline shares `evict_value_v1`'s offline
leave-one-family-out training procedure** — LRB/3L-Cache/CACHEUS adapt
online from their own in-trace stream, HALP trains offline only on each
trace's own history prefix, and LRU/SIEVE/FIFO-Reinsertion are
parameter-free. This is an intentional, disclosed difference in training
mechanics, not an evaluation-protocol mismatch, and it does not make the
comparison unfair (no baseline sees future information or another
family's data — verified per-row).

`baseline_eligibility.csv` (transferred, hash-verified) records, **scoped
to Wulver's own filesystem**:

- `evict_value_v1_cross_family_v1`: `PRIMARY_ELIGIBLE` — this protocol's own artifact.
- `lru`, `sieve`, `fifo_reinsertion`: not found on Wulver under this exact
  window; Wulver's `deployment_full_stream`-only comparison
  (`supplementary_full_stream_comparison.csv`) remains a valid, separately
  caveated, different-run supplementary comparison in its own right, but is
  **not** the only same-protocol evidence available — see the correction
  above.
- `lrb`, `three_l_cache`, `halp`, `cacheus`: not found on Wulver; exact-protocol
  results exist locally on this workstation instead (never synced to
  Wulver).

**Now allowed to claim** (see `major1_reviewer_summary.md` for the full,
validated version): "LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE, and
FIFO-Reinsertion were all evaluated against `evict_value_v1` under an
exact-matched evaluation protocol (same traces, capacities, windows, budget,
and metrics); `evict_value_v1` loses on a clear majority of cells against
every one of them."

**Still prohibited:** "All methods were trained under an identical
procedure" — training mechanics remain algorithm-specific by design.

## Headline result (unchanged from Wulver-side audit, now independently confirmed)

Under the caveated supplementary comparison (`deployment_full_stream`, n=21 cells):
`evict_value_v1` vs `lru` — 0 wins / 18 losses / 3 ties, mean regret +5.29% (95%
bootstrap CI [2.93%, 8.17%], Wilcoxon p=0.000196). See
`heldout_treatment_primary_summary.csv` in this directory for the 21-row primary-window
miss/miss-ratio table, and the canonical `family_summary.csv` /
`capacity_summary.csv` for the by-family / by-capacity regret breakdown (capacity
effect is **not monotonic**: +6.70% / +2.64% / +6.53% at cap 32/64/128).
