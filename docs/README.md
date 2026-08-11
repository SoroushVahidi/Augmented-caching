# Documentation index (`docs/`)

Use this page to find the right document without duplicating long runbooks.

## New to this repository, or a new agent with no conversation history?

| Document | Use when you need… |
|----------|---------------------|
| [`DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md) | **Start here.** Authoritative developer/research handoff: project purpose, current scientific question, branch roles, evidence summary, invalid artifacts, active local work, and a startup checklist |
| [`NEW_AGENT_HANDOFF.md`](NEW_AGENT_HANDOFF.md) | Crisp "what do I do right now" -- immediate next actions (with machine/entry point/stopping condition) and a hard safety "Do Not Do" list |
| [`NEXT_STEPS.md`](NEXT_STEPS.md) | Full P0-P4 roadmap, each item with why/status/dependency/machine/entry point/cost/stopping rule |

## Current KBS second-revision reviewer science

| Document | Use when you need… |
|----------|---------------------|
| [`reviewer/KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md`](reviewer/KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md) | Canonical index of every reviewer-relevant experiment: question, protocol, scope, status, evidence strength, next action |
| [`reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`](reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md) | What answers each reviewer concern, and how completely (`ANSWERED`/`PARTIAL`/`MISSING`/...) |
| [`reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`](reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md) | The mechanistic-hypothesis matrix (H1-H11): evidence, status, decisive next experiment, stopping rule |
| [`reviewer/KBS_SECOND_REVISION_REPRODUCIBILITY.md`](reviewer/KBS_SECOND_REVISION_REPRODUCIBILITY.md) | How to actually rerun/resume a second-revision diagnostic, canonical windows/seeds, checkpoint semantics |
| [`kbs_manuscript_workflow.md`](kbs_manuscript_workflow.md) | Current workflow hub and distinction from historical `heavy_r1` material |
| [`kbs_second_revision_repository_state.md`](kbs_second_revision_repository_state.md) | Canonical-branch intent, tracked-vs-generated boundaries, and the live status of any active revision-science job |
| [`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md) | Concern-by-concern code, outputs, status, caveats, and the running same-target learning-curve diagnostic |
| [`reviewer/kbs_evidence_eligibility.md`](reviewer/kbs_evidence_eligibility.md) | What can be used in primary tables, supporting analysis, or diagnostics only |
| [`reviewer/kbs_negative_results_interpretation.md`](reviewer/kbs_negative_results_interpretation.md) | Internal note on the negative results, including the same-target scalar-vs-pairwise diagnostic distinction |
| [`reviewer/kbs_comparison_fairness_audit.md`](reviewer/kbs_comparison_fairness_audit.md) | Cross-cutting fairness audit of the baseline/method comparison pool |
| [`reviewer_revision_roadmap.md`](reviewer_revision_roadmap.md) | Current status summary for the four reviewer concerns plus supplementary local diagnostics |

## Local-Wulver consolidation

| Document | Use when you need… |
|----------|---------------------|
| [`reviewer/KBS_LOCAL_WULVER_CONFLICT_MATRIX.md`](reviewer/KBS_LOCAL_WULVER_CONFLICT_MATRIX.md) | Per-file conflict-risk classification before any future push/merge with Wulver |
| [`reviewer/KBS_LOCAL_TO_WULVER_MASTER_MANIFEST.md`](reviewer/KBS_LOCAL_TO_WULVER_MASTER_MANIFEST.md) | Whole-branch transfer plan: git payload, living docs, generated results, do-not-transfer |
| [`reviewer/KBS_LOCAL_TO_WULVER_SYNC_STATUS.md`](reviewer/KBS_LOCAL_TO_WULVER_SYNC_STATUS.md) | What's believed already on both sides vs. local-only, at a whole-branch level |
| [`reviewer/local_to_wulver_continuation_sync_manifest.md`](reviewer/local_to_wulver_continuation_sync_manifest.md) | Protocol-specific sync manifest for the C1/C2 continuation work, with pinned source/test/config hashes |

## Historical KBS / Wulver `heavy_r1`

| Document | Use when you need… |
|----------|---------------------|
| [`../CANONICAL_KBS_SUBMISSION.md`](../CANONICAL_KBS_SUBMISSION.md) | Historical heavy-run checklist hub |
| [`kbs_manuscript_workflow.md`](kbs_manuscript_workflow.md) | Current workflow plus historical `heavy_r1` pointers |
| [`evict_value_v1_kbs_canonical_artifacts.md`](evict_value_v1_kbs_canonical_artifacts.md) | Exact `*_heavy_r1` filenames for the older builder path |
| [`kbs_manuscript_submission_index.md`](kbs_manuscript_submission_index.md) | Reviewer-facing index for the older heavy-run line |
| [`wulver_heavy_evict_value_experiment.md`](wulver_heavy_evict_value_experiment.md) | Historical Slurm runbook, defaults, success checks |

## Reproducibility, baselines, and framework

| Document | Use when you need… |
|----------|---------------------|
| [`reproducibility_and_artifacts.md`](reproducibility_and_artifacts.md) | CLI entry points, output locations, manuscript vs exploratory |
| [`repo_map.md`](repo_map.md) | Top-level directory orientation |
| [`baselines.md`](baselines.md) | Baseline policy definitions and literature pointers |
| [`framework.md`](framework.md) | Experimental policy families and architecture notes |
| [`datasets.md`](datasets.md) | Dataset formats and preparation |
| [`datasets_wulver_trace_acquisition.md`](datasets_wulver_trace_acquisition.md) | Wulver trace acquisition notes |

## Evidence strength, open questions, and audits

| Document | Use when you need… |
|----------|---------------------|
| [`manuscript_open_questions.md`](manuscript_open_questions.md) | Priority-ordered research and positioning risks. **Likely superseded/orphaned**: last touched 2026-04, predates the H1-H11 hypothesis map and the second-revision experiment registry, and describes an earlier TIST pairwise-vs-pointwise framing not otherwise referenced by current second-revision docs -- flagged here rather than deleted |
| [`manuscript_evidence_map.md`](manuscript_evidence_map.md) | Claim-by-claim table for older manuscript-support work. Same staleness caveat as above; points to `kbs_manuscript_workflow.md` for the current canonical pipeline |
| [`manuscript_tist_positioning.md`](manuscript_tist_positioning.md) | TIST-oriented positioning notes |

## Exploratory and internal material

| Document | Notes |
|----------|--------|
| [`lightweight_exploratory_ablations.md`](lightweight_exploratory_ablations.md) | Index for `analysis/*_light/` |
| `pairwise_*.md` | Theorem development and audits; not finalized proofs |
| [`offline_general_caching_approx.md`](offline_general_caching_approx.md) | Separate experiment family |
| `internal_*` docs | Author-facing working notes; not canonical evidence |

## Repository hygiene and cleanup

| Document | Role |
|----------|------|
| [`repository_cleanup_report.md`](repository_cleanup_report.md) | Navigation and cleanup notes |
| [`kbs_repository_hygiene_report.md`](kbs_repository_hygiene_report.md) | Earlier KBS hygiene notes |
| [`repo_hygiene_cleanup_report_2026-04-11.md`](repo_hygiene_cleanup_report_2026-04-11.md) | Dated snapshot |
