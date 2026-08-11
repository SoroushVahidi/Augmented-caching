# Wulver-to-GitHub Promotion Queue

Ranked queue of Wulver-only (or Wulver-ahead) items, for when a session with
actual Wulver access is available to sync them back. This workstation does
not contact Wulver; this file only records intent and priority, sourced
from the facts in [`CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md).

**Last reconciled:** 2026-08-11.

Classification vocabulary:

| Label | Meaning |
|---|---|
| `PROMOTE_NOW` | Result/code is complete and stable; sync and (where general-purpose) promote to `main` at the next opportunity |
| `WAIT_FOR_RUNNING_JOB` | Underlying Wulver job is still executing; nothing to sync yet |
| `NEEDS_REVIEW` | Complete, but needs a semantic/fairness read before promotion (e.g. protocol-matching questions) |
| `DO_NOT_PROMOTE` | Explicitly not ready, or blocked by an unresolved defect |
| `SUPERSEDED` | No longer the right artifact to promote; a newer version exists or is coming |

---

## 1. Corrected held-out `evict_value_v1` resume/finalization orchestration

- **What:** the 42/42 corrected cross-family held-out result
  (`analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv`,
  SHA-256 `982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296`)
  plus whatever finalization/orchestration script produced it.
- **Classification:** **`NEEDS_REVIEW`** (not `PROMOTE_NOW`) -- this is the
  single highest-value item, but before treating it as the primary
  reviewer-response table it needs: (a) a hash/row-count/integrity check
  against the criteria already used for the local `25%` learning-curve
  audit (unique keys, no NaN/Inf, all `status=ok`); (b) confirmation the
  model registry it scored against matches the frozen, held-out-eligible
  `evict_value_v1_cross_family_v1` registry already trained locally
  (7/7 folds); (c) rerunning
  `scripts/analysis/prepare_r2_major1_evidence.py --treatment-csv ...` to
  materialize the final matched comparison against the locally validated
  LRB/3L-Cache/CACHEUS controlled-window rows. It does **not** need to wait
  for Wulver jobs `1171965`-`1171967` unless their missing config later
  proves materially different from the local protocol.
- **Why highest priority:** this is the item every other R2 Major 1 gap
  depends on; without it, the method's own headline comparison has no
  valid primary row at all (see `CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`).

## 2. Controlled timing runner/sbatch/config + final analysis provenance

- **What:** Wulver job `1171758`'s 420/420-row raw campaign and its
  analysis/provenance (mean per-request runtimes for LRU/FIFO-
  Reinsertion/SIEVE/HALP-causal).
- **Classification:** **`PROMOTE_NOW`** -- the job is complete, the result
  shape (7 families x 3 capacities x 4 policies x 5 repetitions) matches
  the pre-registered controlled design, and the numbers are internally
  coherent (HALP-causal's ~186x overhead vs. LRU is a plausible, explicit,
  citable finding, not a red flag). Sync the CSV/provenance and the
  `run_practical_significance_controlled.py` config used to produce it;
  the runner code itself is already pushed.
- **Caveat to carry forward on promotion:** wall-clock implementation
  evidence, not an algorithmic complexity theorem; modern LRB/3L/CACHEUS
  timing is not included in this 4-policy campaign and may need a separate
  pass if required for R2 Major 4.

## 3. Broad degeneracy/margin diagnostic source (and job `1169513` result)

- **What:** the 21-cell (7 families x 3 capacities) target-degeneracy
  result and whatever campaign driver produced it at that scale (the local
  `analyze_eviction_loss_target_degeneracy.py` is single-cell-parameterized
  but was run manually cell-by-cell locally; Wulver's driver for the full
  21-cell sweep is not confirmed present in any local worktree).
- **Classification:** **`NEEDS_REVIEW`** -- the *result* is complete and
  important (unique-winner fraction = 0 across all 21 cells is a strong,
  generalizing confirmation of the local single-cell finding), but the
  *driver/orchestration source* for the 21-cell sweep needs to be located
  and reviewed before promotion, since it isn't yet known to exist locally.
  If it turns out to just be the existing single-cell script invoked in a
  loop, promotion is trivial; if Wulver built new orchestration, that code
  needs the same review any other promotion candidate gets.
- **Explicit caveat to carry forward:** the capacity trend (higher capacity
  -> higher zero-margin/optimal-set fractions) is **empirical evidence, not
  a mathematical H/C law** -- do not promote this as proof of the H/C
  framing in `HYPOTHESIS_MAP.md` H10.

## 4. Historical-tail diagnostic source

- **What:** whatever implementation produced Wulver job `1169665`'s result
  (H=8 resolves ~24.6% of H=4 ties, H=16 resolves ~38.7%).
- **Classification:** **`NEEDS_REVIEW`** -- this diagnostic has never
  existed locally at all (confirmed absent by fresh grep this pass), so
  there is no local implementation to compare against; the entire
  source needs first-time review before it can be promoted anywhere,
  including to this branch.
- **Explicit caveat to carry forward:** weak support for horizon/tail
  concerns, not a downstream policy win -- the diagnostic measures tie
  resolution, not miss-ratio improvement.

## 5. HALP causal source/tests/sbatch

- **What:** HALP is already implemented and validated locally
  (`FINAL_VALIDATED`, 42/42, `LOW_TO_MEDIUM` fidelity caveat) -- no new
  Wulver-only HALP source was reported in this reconciliation.
- **Classification:** **`SUPERSEDED`** by the fact that this item is
  already done locally; nothing to promote from Wulver for HALP
  specifically. (Listed here only because the task template anticipated
  it as a candidate -- audit found no gap.)

## 6. Distribution-shift orchestration (merged 24/42 state)

- **What:** the additional ~6 rows behind Wulver's merged 24/42
  distribution-shift state, beyond the local 18/42 checkpoint.
- **Classification:** **`WAIT_FOR_RUNNING_JOB`** in spirit -- not literally
  a running job, but an incremental partial state that isn't worth a
  dedicated sync trip on its own; bundle with the next horizon-sensitivity
  or continuation-fix sync once one of those needs Wulver contact anyway.
- **Caveat to carry forward:** across the 12 paired cells analyzed, misses
  worsened in 9, improved in 0, tied in 3 -- do not claim distribution-
  shift correction solves the online-performance gap; if anything this
  strengthens the existing negative-result narrative.

## 7. Horizon runner/config/sbatch, after job `1169299` finishes

- **What:** the base-horizon sweep (H in {1,2,4,8,16}) driver.
- **Classification:** **`WAIT_FOR_RUNNING_JOB`** -- job `1169299` is at
  17/35 (H=1 and H=2 complete for all families; H=4 complete for
  brightkite/citibike/cloudphysics only; remaining H=4 cells and all
  H=8/H=16 pending). Do not attempt a partial sync/promotion of an
  in-progress sweep; wait for completion.

## 8. Exact-protocol LRB/3L/CACHEUS config/sbatch, after jobs finish

- **What:** the matched-to-corrected-split re-run of LRB (`1171966`),
  3L-Cache (`1171965`), CACHEUS (`1171967`).
- **Classification:** **`WULVER_PENDING_REPLICATION`** -- all three are
  `PENDING`, blocked by Wulver maintenance (not failed), but a fresh local
  audit found complete controlled-window CSVs for all three policies under
  `analysis/reviewer_fairness/`: LRB
  `LOCAL_EXACT_PROTOCOL_VALIDATED`, 3L-Cache
  `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_CAVEAT`, and CACHEUS
  `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_PROVENANCE_CAVEAT`. When Wulver
  returns, promote these jobs only as independent replication or to check
  whether the missing Wulver config JSON contains an additional constraint
  not recorded in local docs/source.

## 9. Continuation causal driver, only after semantic/interface repair

- **What:** `src/lafc/continuation_policy_ablation.py` and its production
  runner.
- **Classification:** **`DO_NOT_PROMOTE`** -- a real defect blocks
  production use: the runner expects a `reference_model=` keyword that the
  protected/pinned source does not provide, causing an unexpected-keyword
  failure. The existing draft also only implements two conditions (of the
  three needed for a full C0/C1/C2 comparison, per the design in
  `KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md` #10). Status must read
  `CONCEPTUAL_BUT_NOT_PRODUCTION_READY`, not `READY_TO_RUN`, until this is
  fixed -- on either machine. This is currently the single largest
  remaining gap for R2 Major 3 / R3's causal-explanation concern.

---

## Priority summary (top 5)

1. **Corrected held-out `evict_value_v1` (item 1)** -- unblocks R2 Major 1's final synthesis; `NEEDS_REVIEW` before use.
2. **Continuation C0/C1/C2 interface repair (item 9)** -- the primary remaining gap for the causal-explanation reviewer concern (R3); `DO_NOT_PROMOTE` until the `reference_model=` mismatch is fixed.
3. **Controlled timing (item 2)** -- complete and ready; `PROMOTE_NOW`.
4. **Broad degeneracy (item 3)** -- important generalizing confirmation of the local single-cell finding; `NEEDS_REVIEW` pending driver-source location.
5. **Exact-protocol LRB/3L/CACHEUS (item 8)** -- replication/provenance strengthening only; not required for R2 Major 1 synthesis unless config audit later finds a material difference.
