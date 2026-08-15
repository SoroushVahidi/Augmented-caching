# Documentation index (`docs/`)

Use this page to find the **right** document without duplicating long runbooks. **Current reviewer gateway:** [`reviewer/START_HERE.md`](reviewer/START_HERE.md). The older Wulver `heavy_r1` checklist is historical: [`../historical/CANONICAL_KBS_SUBMISSION.md`](../historical/CANONICAL_KBS_SUBMISSION.md).

---

## Experimental evidence and scientific organization

**Start here for the science.** These pages are organized by scientific
question, not by manuscript-revision bookkeeping, and are the intended
public entry points.

| Document | Use when you need… |
|----------|---------------------|
| [`EXPERIMENTAL_EVIDENCE.md`](EXPERIMENTAL_EVIDENCE.md) | The scientific questions this project investigates and where the evidence for each stands |
| [`RESULTS_AND_LIMITATIONS.md`](RESULTS_AND_LIMITATIONS.md) | Honest account of what worked, what didn't, and what remains open, including negative results |
| [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) | Canonical index of every experiment: question, scope, status, evidence strength, reproducibility |
| [`HYPOTHESIS_MAP.md`](HYPOTHESIS_MAP.md) | Mechanistic hypotheses for the offline-to-online gap, evidence, and decisive next tests |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Stable install/setup, protocol conventions, and reproduction commands |

---

## Canonical KBS / Wulver `heavy_r1` (main submission line)

| Document | Use when you need… |
|----------|---------------------|
| [`CANONICAL_KBS_SUBMISSION.md`](../historical/CANONICAL_KBS_SUBMISSION.md) | HISTORICAL Wulver `heavy_r1` checklist (not current reviewer evidence) |
| [`kbs_manuscript_workflow.md`](kbs_manuscript_workflow.md) | Full workflow, builder command, separation from exploratory work |
| [`evict_value_v1_kbs_canonical_artifacts.md`](evict_value_v1_kbs_canonical_artifacts.md) | Exact filenames for builder `EVIDENCE_FILES` |
| [`kbs_manuscript_submission_index.md`](kbs_manuscript_submission_index.md) | Reviewer-facing index for the same line |
| [`wulver_heavy_evict_value_experiment.md`](wulver_heavy_evict_value_experiment.md) | Slurm runbook, defaults, success checks |
| [`evict_value_v1_method_spec.md`](evict_value_v1_method_spec.md) | Repository-derived method facts (features, labels, splits, selection) |
| [`method_detail_support_evict_value_v1.md`](method_detail_support_evict_value_v1.md) | Internal consolidation for Methods rewrites (not an artifact) |
| [`kbs_knowledge_framing_note.md`](kbs_knowledge_framing_note.md) | Safe “knowledge-based” framing without over-claiming |
| [`kbs_author_writing_evict_value_v1.md`](kbs_author_writing_evict_value_v1.md) | Author notes tied to KBS artifacts |

**Primary vs duplicate:** `kbs_manuscript_workflow.md` remains the **narrative** workflow hub; `CANONICAL_KBS_SUBMISSION.md` is the **checklist** hub. Cross-links replace merging two long documents.

---

## Reproducibility, baselines, and framework

| Document | Use when you need… |
|----------|---------------------|
| [`reproducibility_and_artifacts.md`](reproducibility_and_artifacts.md) | CLI entry points, output locations, manuscript vs exploratory |
| [`repo_map.md`](repo_map.md) | Top-level directory orientation |
| [`baselines.md`](baselines.md) | Baseline policy definitions and literature pointers |
| [`framework.md`](framework.md) | Experimental policy families and architecture notes |
| [`datasets.md`](datasets.md) | Dataset formats and preparation |
| [`datasets_wulver_trace_acquisition.md`](datasets_wulver_trace_acquisition.md) | Wulver trace acquisition notes |

---

## Evidence strength, open questions, and audits

| Document | Use when you need… |
|----------|---------------------|
| [`manuscript_open_questions.md`](manuscript_open_questions.md) | Priority-ordered research and positioning risks |
| [`manuscript_evidence_map.md`](manuscript_evidence_map.md) | Claim-by-claim table (includes exploratory pairwise line) |
| [`manuscript_tist_positioning.md`](manuscript_tist_positioning.md) | TIST-oriented positioning notes (separate from KBS path) |

---

## Exploratory: pairwise, theory, guards, offline teachers

| Document | Notes |
|----------|--------|
| [`pairwise_vs_pointwise_experiment.md`](pairwise_vs_pointwise_experiment.md) | Controlled comparison; interpret per evidence map |
| `pairwise_*.md` (many files) | Theorem development and audits; not finalized proofs — start from [`pairwise_theory_roadmap.md`](pairwise_theory_roadmap.md) |
| [`wulver_pairwise_publishability_campaign.md`](wulver_pairwise_publishability_campaign.md), [`wulver_pairwise_chain_witness_campaign.md`](wulver_pairwise_chain_witness_campaign.md) | Campaign runbooks |
| [`guarded_robust_wrapper.md`](guarded_robust_wrapper.md) | `evict_value_v1_guarded` specification |
| [`lightweight_exploratory_ablations.md`](lightweight_exploratory_ablations.md) | Index for `analysis/*_light/` |
| [`offline_teacher_*.md`](offline_teacher_vs_heuristic_experiment.md), [`offline_general_caching_approx.md`](offline_general_caching_approx.md) | Separate experiment families |

**Decision-aligned v2 (not v1 heavy supervision):** [`decision_aligned_eviction_targets.md`](decision_aligned_eviction_targets.md), [`decision_aligned_targets.md`](decision_aligned_targets.md).

---

## Internal (`internal_*`) — not canonical evidence

Author-facing working notes, novelty guardrails, bibliography gaps, and prior-work matrices. **Do not** treat as peer-reviewed claims. See filenames under `docs/internal_*`.

---

## Repository hygiene and cleanup

| Document | Role |
|----------|------|
| [`repository_cleanup_report.md`](repository_cleanup_report.md) | What changed in journal-readiness passes (navigation, not science) |
| [`kbs_repository_hygiene_report.md`](kbs_repository_hygiene_report.md) | Earlier KBS hygiene notes |
| [`repo_hygiene_cleanup_report_2026-04-11.md`](repo_hygiene_cleanup_report_2026-04-11.md) | Dated snapshot |
