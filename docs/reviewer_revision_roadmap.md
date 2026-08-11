# KBS second-revision roadmap

This file is the current high-level status record for the four KBS
second-revision reviewer concerns. It supersedes older job-by-job notes that
were tied to transient tmux sessions and mid-run states.

Detailed artifact classification lives in
[`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md).
Evidence-usage rules live in
[`reviewer/kbs_evidence_eligibility.md`](reviewer/kbs_evidence_eligibility.md).
The authoritative per-concern status matrix lives in
[`reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`](reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md);
the mechanistic-hypothesis matrix lives in
[`reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`](reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md).
This file should not duplicate either table.

**2026-08-11 note:** the coverage map above was reconciled against fresh
Wulver-side facts on this date (corrected held-out `evict_value_v1` 42/42
complete, controlled timing 420/420 complete, broad degeneracy 21-cell
result, historical-tail result, continuation C0/C1/C2 interface blocker) --
see [`CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md)
and [`WULVER_TO_GITHUB_PROMOTION_QUEUE.md`](WULVER_TO_GITHUB_PROMOTION_QUEUE.md).
Some concern-status labels below (particularly Concern 4) predate that
reconciliation -- treat the coverage map as authoritative if the two
disagree.

## Status labels

| Label | Meaning |
|---|---|
| `COMPLETE_VALIDATED` | Planned artifact set exists and required audits/gates pass. |
| `PARTIAL` | Some valid evidence exists, but the planned campaign is incomplete. |
| `RUNNING_LOCAL` | A local diagnostic campaign is actively running and must not be treated as final evidence. |
| `DIAGNOSTIC_PARTIAL` | Some immutable diagnostic cells are valid to inspect, but the aggregate remains partial or incomplete. |
| `LOCAL_FOUNDATION_ONLY` | Local code, tests, and documentation exist, but the full scientific replay has not been run yet. |
| `SMOKE_ONLY` | Useful implementation/profiling evidence exists, but not controlled final evidence. |
| `PENDING_CONTROLLED_RUN` | Protocol exists; final timing or eval still needs a controlled run. |
| `CONCEPTUAL_ONLY` | The idea is documented, but no dedicated empirical campaign has been run. |
| `CONTAMINATED_DO_NOT_USE` | Artifact exists for documentation or provenance only; not valid for the target comparison. |
| `HISTORICAL` | Kept for provenance or older manuscript tooling; not the default current path. |

## Current concern status

### Concern 1: learned-baseline comparison

Status: `PARTIAL`

- Complete and reviewer-usable under the primary controlled window:
  LRB, 3L-Cache, HALP, CACHEUS, plus the non-learned baselines already in
  `analysis/reviewer_fairness/`.
- Still incomplete:
  the eligible held-out `evict_value_v1` comparison rows from the frozen
  cross-family retraining protocol.
- Explicitly ineligible:
  `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`.

### Concern 2: supervision-objective ablation

Status: `COMPLETE_VALIDATED`

- 28/28 models complete.
- 84/84 held-out rows complete.
- Same-example audit: `PASS`.
- Fairness audit: `PASS`.
- Registry gate: frozen.

### Concern 3: distribution-shift diagnosis

Status: `PARTIAL`

- The canonical worktree preserves a valid local checkpoint with `24/42` rows.
- The local evidence already shows strong trajectory divergence, but not a full
  seven-family conclusion.

### Concern 4: practical significance

Status: `SMOKE_ONLY`

- Exact-decision-preserving optimization evidence exists.
- Smoke-scale speedups and break-even calculations exist.
- Controlled final timing remains `PENDING_CONTROLLED_RUN`.

## Supplementary local diagnostics

### Same-target scalar-vs-pairwise learning curve

Status: `25_PERCENT_COMPLETE_VALID`; `50%` `RUNNING_LOCAL_RESUME`
(first attempt launched `2026-08-10` in a foreground SSH shell, not tmux;
terminated after ~80 minutes when that SSH session closed, `0/42` rows
committed; relaunched same day in tmux session
`kbs_learning_curve_50pct_20260810`, confirmed healthy; `4/7` folds done
as of a `2026-08-10 21:59` read-only re-check, `metakv` in progress);
`100%` not yet started

- tmux session:
  `kbs_learning_curve_highfrac_20260809` (no longer running; no tmux server
  process present on this host as of the `2026-08-10` audit)
- currently audited low-fraction checkpoint:
  `1%, 2%, 5%, 10%`
- completed families in that audited checkpoint:
  `brightkite, citibike, cloudphysics, metacdn`
- validated units / rows:
  `16` units, `96` rows
- last completed phase:
  `25%`
- current 25% checkpoint:
  `7/7` folds have unit audits and result rows (completed naturally,
  `2026-08-09` `20:17` local time, `7h43m` wall time against a `10`-hour
  budget)
- completed 25% folds:
  `brightkite, citibike, cloudphysics, metacdn, metakv, twemcache, wiki2018`
- local wall-time budget for the `25%` phase:
  `10` hours (not exhausted; job finished naturally)
- later planned phases:
  `50%` (first attempt launched `2026-08-10` outside tmux in a foreground
  SSH shell; terminated after ~80 minutes when that SSH session closed,
  `0/42` rows committed; relaunched same day correctly in tmux session
  `kbs_learning_curve_50pct_20260810` via `--resume --fractions 0.5
  --max-wall-hours 10`, `7`-fold target, clean 10-hour wall-budget stop
  semantics, `RUNNING_LOCAL_RESUME`; `4/7` folds complete as of a
  `2026-08-10 21:59` read-only re-check — `brightkite`, `citibike`,
  `cloudphysics`, `metacdn` done, `metakv` in progress, `twemcache` and
  `wiki2018` remaining; observed per-family runtimes suggest this
  invocation may not finish all `7` folds before its `10`-hour clean-stop
  and a further resume may be needed), then `100%` (not yet launched)
- output:
  `analysis/supervision_objective_learning_curve_v1/`
- no final claim yet; `25%` cell integrity is validated but no aggregate
  learning-curve conclusion has been drawn.

Scientific distinction to preserve:

- earlier `objective_pairwise` results changed the supervision objective itself;
- the new diagnostic compares `eviction_loss_scalar` against
  `eviction_loss_pairwise`, where pairwise labels are derived from the same
  eviction-loss scalar labels and the same sampled decisions.

Current preliminary observation:

- completed `1%`, `2%`, `5%`, and `10%` cells currently favor scalar over the
  same-target pairwise condition,
- this remains preliminary until the active `25%` phase reaches a clean stop
  and completed units are audited,
- later `50%` and `100%` phases remain required for the full learning-
  convergence question.

### Horizon sensitivity

Status: `PENDING_CONTROLLED_RUN`

- planned externally on Wulver, not queried from this workstation,
- requires a v2 protocol and retraining,
- no current regime claim should be made from `H=4` alone.

### Pairwise-cycle diagnostic

Status: `CONCEPTUAL_ONLY`

- the frozen `objective_pairwise` deployment path uses scalar candidate rewards,
  so arbitrary pairwise cycles are absent by construction,
- a real cycle-frequency diagnostic only becomes meaningful for a future
  explicit pairwise comparator.

### Minimum counterfactual attribution

Status: `CONCEPTUAL_ONLY`

- retain the exact DP or shortest-path framing,
- bounded-lookback relaxations remain plausible,
- multiple equally minimal repairs must be preserved,
- trajectory divergence is not itself harmful divergence.

### Future-aware oracle vs learned-online comparison

Status: `SMOKE_VALIDATED / ONE-CELL REAL-TRACE DIAGNOSTIC COMPLETE` and `HIGH_PRIORITY`

- local foundation code now exists in `src/lafc/oracle_diagnostics.py`,
  reusing the exact target kernel in
  `src/lafc/supervision_objective_ablation.py`;
- focused synthetic validation exists in `tests/test_oracle_diagnostics.py`;
- this diagnostic separates:
  exact target oracle,
  learned online approximation,
  and global offline oracle context from `offline_belady`;
- for `eviction_loss`, the exact target oracle must use the same frozen label
  semantics as training:
  admit the incoming page, evaluate each candidate with finite horizon `H`,
  and continue the label rollout under `LRU`;
- later full evaluation should report:
  exact-oracle misses,
  learned-policy misses,
  Belady misses,
  learned-vs-exact decision agreement,
  exact-target regret,
  and downstream miss gap;
- first local real-trace diagnostic completed for
  `brightkite`, capacity `64`, horizon `4`, canonical controlled window
  `[10000,50000)`;
- output path:
  `analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/`;
- result on this single cell:
  LRU `13225` misses,
  exact finite-horizon eviction-loss oracle `19079` misses,
  learned eviction-loss scalar policy `15449` misses,
  offline Belady `11312` misses;
- learned model was provenance-eligible:
  frozen `objective_eviction_loss/brightkite.pkl`, held-out family excluded
  from training, validation family `citibike`, model hash matched registry;
- runtime emitted sklearn model-load compatibility warnings:
  artifact serialized with sklearn `1.9.0`, local runtime `1.8.0`;
- planned horizon grid for this diagnostic:
  `1, 2, 4, 8, 16`;
- do not interpret this one-cell output as a horizon sweep or family-general
  conclusion; Wulver's horizon study and the local learning-curve campaign
  remain separate;
- do not collapse oracle context into deployable-baseline claims;
- this remains separate from the running same-target learning-curve work and
  from minimum-counterfactual suffix attribution.

### Target-degeneracy diagnostic

Status: `ONE_CELL_DIAGNOSTIC_COMPLETE / FULL_SWEEP_NOT_RUN` and `HIGH_PRIORITY`

- local diagnostic code:
  `src/lafc/target_degeneracy.py`;
- local runner:
  `scripts/experiments/analyze_eviction_loss_target_degeneracy.py`;
- focused validation:
  `tests/test_target_degeneracy.py`;
- completed local artifact:
  `analysis/eviction_loss_target_degeneracy_v1/brightkite_cap64_h4/`;
- cell:
  `brightkite`, capacity `64`, base horizon `4`, canonical score window;
- observed in this cell:
  all `19079` H=4 scored decisions have ordinary zero margin,
  `63.0%` have all candidates tied, and median optimal-set size is `64`;
- longer-horizon tie resolution:
  H=8 breaks `14.2%`, H=16 breaks `27.6%`, H=32 breaks `39.6%` of H=4 ties;
- interpretation:
  H=4 target degeneracy/tie saturation is strong in this cell, and longer
  horizons increase resolution but leave most H=4 tied sets unresolved;
- caveat:
  do not generalize across workloads until additional cells are run.

### Continuation-policy causal ablation

Status: `LOCAL_IMPLEMENTATION_READY / PROTOCOL_FROZEN / FULL_RUN_PENDING_WULVER`
and `HIGH_PRIORITY`

- precise condition names are frozen in
  `configs/continuation_policy_causal_ablation_v1.json`;
- current method is interpreted as:
  `pi0 = LRU`, labels use `Q_H^{pi0}`, train `pi1`, deploy recursive `pi1`;
- the new diagnostic tests one continuation-update step:
  labels use `Q_H^{pi1}` with frozen eligible `pi1`, train `pi2`, deploy
  recursive `pi2`;
- implementation source:
  `src/lafc/continuation_policy_ablation.py`;
- focused validation:
  `tests/test_continuation_policy_ablation.py`;
- tiny local smoke runner:
  `scripts/experiments/run_continuation_policy_causal_ablation_smoke.py`;
- C1 vs C2 must use the same decision IDs and candidate IDs; the only intended
  changed variable is label continuation policy (`LRU -> frozen pi1`);
- frozen `pi1` must come from
  `analysis/supervision_objective_ablation_v1/model_registry.json`, objective
  `objective_eviction_loss`, with matching artifact hash and no held-out
  leakage;
- no full result exists yet, and no novelty or performance-improvement claim
  should be made before Wulver execution and audit;
- reviewer linkage:
  R2 Major 3 / R3 continuation-mismatch and policy-iteration interpretation.

### Practical timing

Status: `PENDING_CONTROLLED_RUN`

- smoke-scale exact-optimization and break-even outputs exist,
- controlled final timing remains pending.

## Current unfinished reviewer-target work

Preserve these as active TODO items:

1. finish and audit the `25%` learning-convergence phase (done, `VALID`);
2. `50%` first attempt was `INTERRUPTED_BEFORE_FIRST_COMPLETED_UNIT`
   (launched `2026-08-10` outside tmux, killed when the SSH session
   closed, `0/42` rows); relaunched same day in tmux session
   `kbs_learning_curve_50pct_20260810`, now `RUNNING_LOCAL_RESUME` —
   audit on clean stop;
3. later run `100%`;
4. jointly analyze downstream misses, scalar `MAE/RMSE`, and ranking or
   decision metrics versus fraction;
5. determine whether scalar performance converges with more data or remains
   limited despite improved offline prediction;
6. keep horizon sensitivity explicitly external to this workstation;
7. run and audit the designed exact-target-oracle vs learned-online diagnostic;
8. synchronize and run the continuation-policy causal ablation on Wulver;
9. complete controlled final timing;
10. run the pairwise margin/noise diagnostic if same-target pairwise remains
   weak;
11. preserve the minimum-counterfactual or minimum-Hamming-distance suffix
    attribution line;
12. treat the final R2 Major 1 held-out audit as the next reviewer-target once
    the external Wulver work completes.

### Distribution shift

Status: `PARTIAL`

- the local partial checkpoint remains incomplete,
- current local evidence is diagnostic only, not a final all-family claim.

## What changed relative to older roadmap entries

Older versions of this roadmap tracked active tmux jobs, in-flight partial row
counts, and provisional next steps during live experiment execution. Those
details were useful operationally, but they are now historical. The current
source of truth is:

- repository state:
  [`kbs_second_revision_repository_state.md`](kbs_second_revision_repository_state.md)
- artifact status and caveats:
  [`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md)
- evidence eligibility:
  [`reviewer/kbs_evidence_eligibility.md`](reviewer/kbs_evidence_eligibility.md)
- internal scientific interpretation:
  [`reviewer/kbs_negative_results_interpretation.md`](reviewer/kbs_negative_results_interpretation.md)
