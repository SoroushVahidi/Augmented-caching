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

## Protocol caveat — preserved from the Wulver-side audit, verified still accurate

No baseline of any kind (`LRU`, `SIEVE`, `FIFO-Reinsertion`, `LRB`, `3L-Cache`, `HALP`,
`CACHEUS`) has been computed under this identical corrected leave-one-family-out
protocol. `baseline_eligibility.csv` (transferred, hash-verified) records:

- `evict_value_v1_cross_family_v1`: `PRIMARY_ELIGIBLE` — this protocol's own artifact.
- `lru`, `sieve`, `fifo_reinsertion`: `NOT_ELIGIBLE` for the primary table;
  `ELIGIBLE_WITH_CAVEAT` only for a window-matched (`deployment_full_stream`,
  50,000-request) but **different-protocol-run** supplementary comparison
  (`supplementary_full_stream_comparison.csv`).
- `lrb`, `three_l_cache`, `halp`, `cacheus`: `NOT_AVAILABLE` — implemented in source,
  zero executed result artifacts exist anywhere on Wulver under any protocol.

**Do not claim** "LRB/3L-Cache/HALP/CACHEUS were all compared against
`evict_value_v1` under an identical executed protocol" — that comparison does not
exist. Only `evict_value_v1` itself has true 42/42 same-protocol coverage in this
package; the supplementary LRU/SIEVE/FIFO numbers are a caveated, different-run,
window-matched comparison, and the four modern learned baselines have no results at
all.

## Headline result (unchanged from Wulver-side audit, now independently confirmed)

Under the caveated supplementary comparison (`deployment_full_stream`, n=21 cells):
`evict_value_v1` vs `lru` — 0 wins / 18 losses / 3 ties, mean regret +5.29% (95%
bootstrap CI [2.93%, 8.17%], Wilcoxon p=0.000196). See
`heldout_treatment_primary_summary.csv` in this directory for the 21-row primary-window
miss/miss-ratio table, and the canonical `family_summary.csv` /
`capacity_summary.csv` for the by-family / by-capacity regret breakdown (capacity
effect is **not monotonic**: +6.70% / +2.64% / +6.53% at cap 32/64/128).
