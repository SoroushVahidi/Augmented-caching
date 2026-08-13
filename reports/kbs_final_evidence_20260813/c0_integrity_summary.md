# C0/C1/C2 Continuation-Policy Causal Ablation — Integrity Summary

Canonical raw source: `analysis/continuation_policy_causal_ablation_production_v1/`
(`policy_comparison.csv`, `label_agreement.csv`, `training_summary.csv`,
`unit_completion_manifest.json`, `integrity_summary.json`, `config_snapshot.json`,
`provenance.json`). This file is a compact summary only; the CSVs above remain
the canonical scientific record.

Audit performed: 2026-08-13, read-only, formal post-completion integrity pass
(independent recount, not a re-run of `integrity_summary.json`'s self-report).

## Completeness

- 21/21 expected family×capacity units present, no missing/duplicate/partial/`.tmp-*` units.
- `policy_comparison.csv`: 63/63 rows (21 units × 3 conditions), all `status=ok`.
- `label_agreement.csv`: 21/21 rows.
- `training_summary.csv`: 21/21 rows.
- No duplicate `(unit_id, condition)` / `unit_id` keys.
- No NaN/Inf in any numeric field; `hits + misses == scored_requests` for all 63 rows.
- SHA-256 of every one of the 63 per-unit files matches `unit_completion_manifest.json` exactly.

## Protocol integrity

- History/score windows (`0, 10000, 10000, 50000`), `horizon=4`, `seed=0` uniform across all 63 rows.
- Canonical `trace_sha256` uniform per family.
- `config_snapshot.json`'s `C0_BASELINE_LRU` / `C1_LRU_CONTINUATION_LEARNED_PI1` /
  `C2_PI1_CONTINUATION_LEARNED_PI2` definitions match the intended design; the only
  changed variable between C1 and C2 is `label_continuation_policy: LRU -> frozen_pi1`.
- `leakage_gate` and `same_example_gate` both PASS for all 21 units.

## Model integrity

- pi1 hash unique per family (7 distinct), never shared across families.
- pi2 hash unique per unit (21/21 distinct — no accidental cross-fold reuse).
- All 7 pi1 + 21 pi2 model files verified present on disk with SHA-256 matching the manifest.
- Frozen pi1 registry (`analysis/supervision_objective_ablation_v1/model_registry.json`)
  confirms `MODEL_SELECTION_FROZEN=true`.

## Source-SHA reconciliation

Two historical `source_sha` values appear across units: `a813617f...` (8 units) and
`12798d8482...` (13 units, including all 5 resumed units). Reconciled as
**provenance-only, scientifically equivalent**:

- `git diff --stat a813617 12798d8` touches only `docs/`, `.gitignore`, and
  reviewer-prep files — zero scientific code/config changed.
- The runner's `_check_existing_output()` fail-closed gate re-verifies scientific-snapshot
  equality on every invocation and raises before writing any output on mismatch; all 21
  units completed cleanly, which is only possible if this gate passed on every historical
  invocation.
- Commit `8421167` ("fix: preserve continuation resume provenance") makes resumed units
  correctly inherit the frozen `source_sha_at_runner_start` rather than the live HEAD at
  resume time, guarded by the same scientific-snapshot equality check.

**Not a `PROVENANCE_PROTOCOL_MISMATCH`.**

## Metadata-path integrity

No stale `.tmp-*` or staging-path references found anywhere in campaign metadata
(manifest text scan + hidden-file scan under the campaign directory).

## Overall classification

**FINAL_VALIDATED.**

Caveat: `source_tree_dirty=True` on 10/21 units, traced to unscoped `git status --porcelain`
over this shared multi-experiment repository (reflects concurrent unrelated work elsewhere
in the tree, not this campaign) — non-blocking given the code-level protocol-drift gate above.

## Scientific-result headline (see `c0_continuation_summary.csv` for full 21-row table)

- C2 improves over C1: 13/21 cells; ties: 3/21 (all Wiki2018, degenerate 100%-miss cells);
  worsens: 5/21.
- Macro mean C2−C1 miss-ratio delta: ≈ −0.0102.
- Aggregate misses: C0 = 565,126, C1 = 601,569, C2 = 592,970.
- Strongest improvement: `metacdn` (cap32/64). Strongest counter-example: `brightkite` cap32
  (+0.2433, the single largest effect in the table, opposite direction).

**H5 (continuation-policy mismatch) disposition: `PARTIALLY_SUPPORTED`** — mixed/regime-dependent,
not universally causal. See `mechanistic_hypothesis_summary.md`.
