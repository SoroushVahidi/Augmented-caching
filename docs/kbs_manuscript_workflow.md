# KBS second-revision workflow

This is the current workflow hub for the repository's KBS second-revision
science branch. It replaces the old top-level assumption that the historical
Wulver `heavy_r1` path is the primary orientation for outside readers.

## 1. Start here

Read these in order:

1. [`kbs_second_revision_repository_state.md`](kbs_second_revision_repository_state.md)
2. [`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md)
3. [`reviewer/kbs_evidence_eligibility.md`](reviewer/kbs_evidence_eligibility.md)
4. [`reviewer_revision_roadmap.md`](reviewer_revision_roadmap.md)
5. [`reviewer/kbs_negative_results_interpretation.md`](reviewer/kbs_negative_results_interpretation.md)

These files distinguish:

- tracked source vs generated evidence,
- complete vs partial vs smoke-only reviewer outputs,
- current local evidence vs historical heavy-run material,
- empirical findings vs interpretation hypotheses.

## 2. Current reviewer-science concerns

### Concern 1: learned-baseline comparison

- Frozen protocol and fairness rules:
  [`reviewer_fairness_protocol.md`](reviewer_fairness_protocol.md),
  [`reviewer_fairness_cross_family_v1.md`](reviewer_fairness_cross_family_v1.md)
- Current artifact map entry:
  [`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md)
- Read-only state check:
  `python3 scripts/validation/revision_status.py`

Current local state:

- LRB, 3L-Cache, HALP, and CACHEUS controlled-window comparison rows exist.
- The old `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`
  remains contaminated and ineligible for the primary reviewer table.
- Cross-family held-out `evict_value_v1` retraining is complete enough to
  freeze a registry, but the held-out evaluation rows still remain pending.

### Concern 2: supervision-objective ablation

- Frozen protocol:
  [`supervision_objective_ablation_protocol.md`](supervision_objective_ablation_protocol.md)
- Current artifact map entry:
  [`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md)

Current local state:

- all 28 models exist,
- 84 held-out rows exist,
- same-example audit passed,
- fairness audit passed,
- registry freeze passed.
- preserve the distinction between:
  - `objective_pairwise` in the frozen objective ablation,
  - `eviction_loss_pairwise` in the local same-target learning-curve
    diagnostic.

### Concern 3: distribution-shift diagnosis

- Frozen protocol:
  [`distribution_shift_ablation_protocol.md`](distribution_shift_ablation_protocol.md)
- Current artifact map entry:
  [`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md)

Current local state:

- the canonical worktree preserves a valid partial local checkpoint
  (`24/42` primary rows, `4/7` families complete),
- the missing Wulver family-run orchestration files have not yet been synced
  back to this local branch,
- local evidence supports trajectory-divergence diagnostics, but not a final
  seven-family causal conclusion.

### Concern 4: practical significance

- Frozen protocol:
  [`practical_significance_ablation_protocol.md`](practical_significance_ablation_protocol.md)
- Current artifact map entry:
  [`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md)

Current local state:

- smoke-scale exact-optimization and break-even outputs exist,
- controlled final timing does not yet exist,
- timing conclusions must stay labeled `SMOKE_ONLY` or
  `PENDING_CONTROLLED_RUN`.

## 2.1 Supplementary local diagnostic: same-target learning curve

- runner:
  `scripts/experiments/run_supervision_objective_learning_curve.py`
- config:
  `configs/supervision_objective_learning_curve_v1.json`
- status:
  local `RUNNING_LOCAL` / `DIAGNOSTIC_PARTIAL`
- output:
  `analysis/supervision_objective_learning_curve_v1/`
- models:
  `models/supervision_objective_learning_curve_v1/`

Purpose:

- compare scalar regression on eviction-loss labels against same-target
  pairwise labels derived from those same eviction-loss labels,
- test whether pairwise representation alone looks more sample-efficient once
  the underlying target notion is held fixed.

Use rule:

- completed cells are useful explanatory diagnostics,
- incomplete aggregates are not final manuscript evidence,
- do not pool these results with the earlier `objective_pairwise` ablation as
  though they were the same scientific condition.

## 3. Read-only audit helpers

The repository now includes lightweight validation entry points under
`scripts/validation/`:

- `python3 scripts/validation/revision_status.py`
- `python3 scripts/validation/revision_readiness.py`

These tools do not launch experiments or modify artifacts. They summarize the
current local state across worktrees and point to the real artifact roots that
still hold preserved evidence.

## 4. Historical `heavy_r1` material

The earlier Wulver `heavy_r1` path is still preserved because it underlies
older manuscript-support builders and provenance notes, but it is now treated
as historical for this cleanup pass.

- Historical runbook:
  [`wulver_heavy_evict_value_experiment.md`](wulver_heavy_evict_value_experiment.md)
- Historical artifact set:
  [`evict_value_v1_kbs_canonical_artifacts.md`](evict_value_v1_kbs_canonical_artifacts.md)
- Historical checklist hub:
  [`../CANONICAL_KBS_SUBMISSION.md`](../CANONICAL_KBS_SUBMISSION.md)

Use those documents when you need to understand the older heavy-run line or the
builder-facing `*_heavy_r1` filenames, not as the default orientation for the
current reviewer-science branch.

## 5. Manuscript integration rules

Before manuscript text is refreshed:

- use [`reviewer/kbs_evidence_eligibility.md`](reviewer/kbs_evidence_eligibility.md)
  to avoid mixing primary controlled-window rows with `deployment_full_stream`
  rows,
- treat smoke-scale timing outputs and partial distribution-shift outputs as
  non-final,
- keep the negative-results interpretation note separate from manuscript prose.

The repository note
[`reviewer/kbs_negative_results_interpretation.md`](reviewer/kbs_negative_results_interpretation.md)
is an internal scientific notebook for later writing, not manuscript text.
