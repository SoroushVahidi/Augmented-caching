# KBS Final Local Evidence Package — 2026-08-13

Compact, tracked summary of the local campaigns that closed out the second-revision
mechanistic investigation and evidence gathering: **C0/C1/C2 continuation-policy causal
ablation**, **distribution-shift ablation**, the **corrected held-out `evict_value_v1`
treatment**, and the **controlled timing campaign** (the latter two synced from Wulver
on 2026-08-13). All four are classified **`FINAL_VALIDATED`** (local campaigns) or
**`FINAL_VALIDATED_SYNCED`** (Wulver-synced payloads).

## Purpose

This directory exists so a reader does not have to open the full raw campaign trees to
get the validated numbers, the integrity result, and the hypothesis/reviewer
dispositions those numbers support. It is a summary layer, not a replacement.

## Source campaign directories (canonical, not duplicated here)

- `analysis/continuation_policy_causal_ablation_production_v1/` — C0/C1/C2 raw CSVs,
  per-unit manifests, model registry cross-references.
- `analysis/distribution_shift_ablation_v1/` — distribution-shift raw CSVs, campaign
  state, protocol snapshot.
- `analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/` —
  corrected held-out `evict_value_v1` treatment (42/42), synced from Wulver
  2026-08-13.
- `analysis/kbs_controlled_timing_20260810/` and
  `analysis/kbs_controlled_timing_final_analysis_20260811/` — controlled timing
  campaign (420/420), synced from Wulver 2026-08-13.

All four directories remain the canonical scientific record. Nothing in this package
overrides them; if a number here and a number in the raw CSVs ever disagree, the raw
CSV is authoritative and this package is stale and should be regenerated.

## Validation status

- C0/C1/C2: `FINAL_VALIDATED` (21/21 units, 63/63 policy rows, 21/21 label-agreement,
  21/21 training-summary rows, all integrity gates PASS).
- Distribution-shift: `FINAL_VALIDATED` (7/7 folds, 21/21 paired cells, 42/42 primary,
  42/42 state-shift, 21/21 trajectory rows, all integrity gates PASS, including an
  independent re-run of the canonical `scripts/experiments/audit_distribution_shift_completion.py`
  → `COMPLETE_VALID`).
- Corrected held-out `evict_value_v1` treatment: `FINAL_VALIDATED_SYNCED` (42/42 rows,
  16/16 transfer hashes PASS, all local re-audit gates PASS — see
  `heldout_treatment_integrity.md`).
- Controlled timing campaign: `FINAL_VALIDATED_SYNCED` (420/420 rows, 13/13 transfer
  hashes PASS, all local re-audit gates PASS, policy means independently recomputed
  and matched — see `controlled_timing_integrity.md`).

## Eligibility rules

- Only fields read directly from the canonical raw CSVs/JSON listed above are reported
  here. No scientific value in this package was computed by re-deriving or re-running
  anything — all summary CSVs are straight reshapes/joins of already-validated rows.
- The corrected held-out treatment does **not** establish a same-protocol comparison
  against modern learned baselines (LRB/3L-Cache/HALP/CACHEUS have zero executed
  results under any protocol) or even against LRU/SIEVE/FIFO under the *same* protocol
  run (only a caveated, window-matched, different-run supplementary comparison
  exists). See `heldout_treatment_integrity.md` for the full caveat.
- The controlled timing campaign covers exactly LRU/FIFO-Reinsertion/SIEVE/HALP-causal
  under a 5-repetition protocol; `evict_value_v1`'s runtime is a separate, single-run
  measurement and must never be placed in that 4-policy table. See
  `controlled_timing_interpretation.md`.
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
| `heldout_treatment_primary_summary.csv` | 21 rows, `primary_controlled_window` family×capacity cells: scored requests, misses, miss ratio, fold ID |
| `heldout_treatment_integrity.md` | Transfer hash result, independent local re-audit gate table, protocol/baseline-eligibility caveat, headline regret result |
| `heldout_treatment_provenance.md` | Wulver source path, transfer commands, full hash manifest, historical-contamination provenance note |
| `controlled_timing_summary.csv` | 4-policy compact table: mean/median/stddev/95% CI µs-per-request, slowdown vs. LRU |
| `controlled_timing_integrity.md` | Transfer hash result, independent local re-audit gate table, recomputed policy means, `timing_summary.csv` preliminary-snapshot caveat |
| `controlled_timing_interpretation.md` | Why `evict_value_v1` is excluded from the 4-policy controlled table; claim-safety rules |
| `reviewer_mapping.md` | Reviewer #2 Major 1–4 and Reviewer #3 status table, with exact answers for Major 3 and Reviewer #3 |

## Wulver evidence — synchronization status

Both previously pending Wulver payloads (corrected held-out treatment, controlled
timing) were synced and locally re-audited on 2026-08-13. See `reviewer_mapping.md` →
"Remaining external blockers" — as of this update, no synchronization blockers remain;
the remaining work is manuscript/rebuttal synthesis, not new evidence collection.
