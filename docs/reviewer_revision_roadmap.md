# KBS second-revision roadmap

This file is the current high-level status record for the four KBS
second-revision reviewer concerns. It supersedes older job-by-job notes that
were tied to transient tmux sessions and mid-run states.

Detailed artifact classification lives in
[`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md).
Evidence-usage rules live in
[`reviewer/kbs_evidence_eligibility.md`](reviewer/kbs_evidence_eligibility.md).

## Status labels

| Label | Meaning |
|---|---|
| `COMPLETE_VALIDATED` | Planned artifact set exists and required audits/gates pass. |
| `PARTIAL` | Some valid evidence exists, but the planned campaign is incomplete. |
| `RUNNING_LOCAL` | A local diagnostic campaign is actively running and must not be treated as final evidence. |
| `DIAGNOSTIC_PARTIAL` | Some immutable diagnostic cells are valid to inspect, but the aggregate remains partial or incomplete. |
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

Status: `RUNNING_LOCAL` and `DIAGNOSTIC_PARTIAL`

- tmux session:
  `kbs_learning_curve_highfrac_20260809`
- currently audited low-fraction checkpoint:
  `1%, 2%, 5%, 10%`
- completed families in that audited checkpoint:
  `brightkite, citibike, cloudphysics, metacdn`
- validated units / rows:
  `16` units, `96` rows
- current live phase:
  `25%`
- current live target:
  all seven held-out folds
- local wall-time budget for the active `25%` phase:
  `10` hours
- later planned phases:
  `50%`, then `100%`
- output:
  `analysis/supervision_objective_learning_curve_v1/`
- no final claim yet.

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

Status: `CONCEPTUAL_ONLY` and `HIGH_PRIORITY`

- exact future-aware or oracle-style comparison against learned-online behavior
  remains a high-priority unfinished diagnostic,
- do not collapse oracle context into deployable-baseline claims,
- this remains separate from the running same-target learning-curve work.

### Practical timing

Status: `PENDING_CONTROLLED_RUN`

- smoke-scale exact-optimization and break-even outputs exist,
- controlled final timing remains pending.

## Current unfinished reviewer-target work

Preserve these as active TODO items:

1. finish and audit the `25%` learning-convergence phase;
2. later run `50%`;
3. later run `100%`;
4. jointly analyze downstream misses, scalar `MAE/RMSE`, and ranking or
   decision metrics versus fraction;
5. determine whether scalar performance converges with more data or remains
   limited despite improved offline prediction;
6. keep horizon sensitivity explicitly external to this workstation;
7. complete the future-aware or oracle-vs-learned-online comparison;
8. complete controlled final timing;
9. run the pairwise margin/noise diagnostic if same-target pairwise remains
   weak;
10. preserve the minimum-counterfactual or minimum-Hamming-distance suffix
    attribution line;
11. treat the final R2 Major 1 held-out audit as the next reviewer-target once
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
