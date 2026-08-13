# KBS Final Local Evidence Package — 2026-08-13

Compact, tracked summary of the two local campaigns that closed out the second-revision
mechanistic investigation: **C0/C1/C2 continuation-policy causal ablation** and
**distribution-shift ablation**. Both passed a formal, read-only post-completion
integrity audit on 2026-08-13 and are classified **`FINAL_VALIDATED`**.

## Purpose

This directory exists so a reader does not have to open the full raw campaign trees to
get the validated numbers, the integrity result, and the hypothesis/reviewer
dispositions those numbers support. It is a summary layer, not a replacement.

## Source campaign directories (canonical, not duplicated here)

- `analysis/continuation_policy_causal_ablation_production_v1/` — C0/C1/C2 raw CSVs,
  per-unit manifests, model registry cross-references.
- `analysis/distribution_shift_ablation_v1/` — distribution-shift raw CSVs, campaign
  state, protocol snapshot.

Both directories remain the canonical scientific record. Nothing in this package
overrides them; if a number here and a number in the raw CSVs ever disagree, the raw
CSV is authoritative and this package is stale and should be regenerated.

## Validation status

- C0/C1/C2: `FINAL_VALIDATED` (21/21 units, 63/63 policy rows, 21/21 label-agreement,
  21/21 training-summary rows, all integrity gates PASS).
- Distribution-shift: `FINAL_VALIDATED` (7/7 folds, 21/21 paired cells, 42/42 primary,
  42/42 state-shift, 21/21 trajectory rows, all integrity gates PASS, including an
  independent re-run of the canonical `scripts/experiments/audit_distribution_shift_completion.py`
  → `COMPLETE_VALID`).

## Eligibility rules

- Only fields read directly from the canonical raw CSVs/JSON listed above are reported
  here. No scientific value in this package was computed by re-deriving or re-running
  anything — all summary CSVs are straight reshapes/joins of already-validated rows.
- Two items remain `SYNC_PENDING` and are explicitly **not** included as validated
  local evidence: corrected held-out `evict_value_v1` treatment (42/42) and controlled
  timing (420/420), both Wulver-only pending synchronization and audit.
- Model-binary retention: 4 of 7 families' `.pkl` files for the distribution-shift
  campaign are no longer present on disk (see `distribution_integrity_summary.md`,
  `MODEL_BINARY_RETENTION_GAP`). Their recorded hashes are internally consistent but
  could not be independently re-verified against a live binary in this pass — do not
  cite full binary reproducibility for those four families.

## Contents

| File | Role |
|---|---|
| `c0_continuation_summary.csv` | 21 rows, one per family×capacity cell: C0/C1/C2 misses, miss ratios, pairwise deltas, label agreement |
| `c0_integrity_summary.md` | Full integrity-gate table, source-SHA provenance reconciliation, H5 headline |
| `distribution_shift_summary.csv` | 21 rows, one per family×capacity cell: OFF/DAGGER misses, miss ratios, delta, state-shift indices, trajectory-divergence metrics |
| `distribution_integrity_summary.md` | Full integrity-gate table, regenerated completion-audit result, log-ordering diagnosis, model-binary retention caveat, H6 headline |
| `mechanistic_hypothesis_summary.md` | Final H1–H9 disposition table, including the two hypotheses closed by this evidence (H5, H6) |
| `reviewer_mapping.md` | Reviewer #2 Major 1–4 and Reviewer #3 status table, with exact answers for Major 3 and Reviewer #3 |

## Wulver evidence still pending

See `reviewer_mapping.md` → "Remaining external blockers." Both are synchronization-only;
no new local experiment is required to close either.
