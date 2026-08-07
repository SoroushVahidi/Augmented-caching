# Repository map

This document is a concise orientation guide for external readers and manuscript reviewers.

## KBS manuscript workflow (submission path)

For **Knowledge-Based Systems** and the canonical Wulver **`heavy_r1`** `evict_value_v1` line (Slurm drivers, evidence files, `build_kbs_main_manuscript_artifacts.py`, `tables/manuscript/`, `figures/manuscript/`):

- **Checklist (scripts, paths, do-not-cite):** [`../CANONICAL_KBS_SUBMISSION.md`](../CANONICAL_KBS_SUBMISSION.md)
- **Narrative workflow:** `docs/kbs_manuscript_workflow.md`
- **Reviewer index:** `docs/kbs_manuscript_submission_index.md`
- **All documentation index:** `docs/README.md`

## Top-level layout

- `src/lafc/` — core library implementation (policies, simulator, runners, datasets, offline solvers).
- `scripts/` — structured experiment and setup entry points (see `scripts/README.md`).
- `configs/` — experimental configurations and protocols (see `configs/README.md`).
- `tests/` — unit/integration tests for policies, datasets, runners, and experiments.
- `docs/` — method notes, experiment protocols, theorem-development notes, and manuscript-support docs.
- `analysis/` — generated text artifacts from experiments (`.csv`, `.json`, `.md`).
- `data/` — small examples in git + raw/processed derived data roots.
- `models/` — trained model artifacts and staging areas.
- `artifacts/` — final manuscript submission packages and external releases.
- `slurm/` — cluster batch templates for heavier runs.

## `src/lafc/` subpackages

- `policies/` — policy implementations (baseline, robust, and experimental).
- `simulator/` — cache state and request trace execution logic.
- `runner/` — CLI entrypoint (`python -m lafc.runner.run_policy`).
- `datasets/` — dataset ingestion, preprocessing, and CLI glue.
- `offline/` — offline reference solvers and IO/helpers.
- `learned_gate/` — v1/v2 learned gate datasets, features, and models.
- top-level `evict_*` / `offline_teacher_supervision.py` — eviction-learning datasets/models and supervision helpers.
- `metrics/` — common cost and prediction error metrics.

## `scripts/` organization

- `scripts/setup/` — dataset ingestion, download, and preprocessing.
- `scripts/maintenance/` — search, theorem-support, and aggregation utilities.
- `scripts/validation/` — revision readiness and status runners.
- `scripts/experiments/canonical/` — manuscript-safe canonical Wulver pipeline.
- `scripts/experiments/reviewer/` — reviewer-revision experiments.
- `scripts/experiments/exploratory/` — exploratory lightweight ablations.
- `scripts/experiments/diagnostics/` — failure-slice audits and diagnostic tools.

## `analysis/` organization

- `analysis/manuscript_canonical/` — frozen results for the final KBS manuscript.
- `analysis/reviewer_revision/` — results from reviewer-revision experiments.
- `analysis/exploratory/` — first-check runners and ad-hoc ablations.
- `analysis/diagnostics/` — failure-slice audits and benchmarks.
- `analysis/manifests/` — trace manifests and experiment split definitions.

See `analysis/README.md` for details and naming guidance.

## Manuscript-support docs to read first

1. `docs/kbs_manuscript_workflow.md` (canonical `heavy_r1` path + builder + outputs + “not canonical” pointers)
2. `docs/evict_value_v1_kbs_canonical_artifacts.md` (heavy_r1-only inputs for KBS tables/figures)
3. `docs/reproducibility_and_artifacts.md` (entry points, output locations, manuscript vs exploratory)
4. `docs/kbs_manuscript_submission_index.md` (reviewer-facing index)
5. `docs/lightweight_exploratory_ablations.md` (non-canonical lightweight branch index)
6. `docs/manuscript_evidence_map.md`
7. `docs/manuscript_open_questions.md`
8. `docs/baselines.md`
9. `docs/framework.md`
10. `docs/internal_research_summary_eviction_value.md` (internal working-state note; not manuscript text, not canonical evidence)
11. `docs/internal_prior_work_audit_eviction_value.md` (internal prior-work coverage + bibliography-gap audit; non-canonical)
12. `docs/internal_current_project_decisions.md` (internal record of current agreed framing decisions; non-canonical)
13. `docs/internal_novelty_positioning_eviction_value.md` (internal novelty-positioning guardrails for related-work and claim scope; non-canonical)
14. `docs/internal_prior_work_matrix_eviction_value.md` (internal comparison matrix for eviction-value related-work and safest novelty scope; non-canonical)
15. `docs/internal_bibliography_gap_report.md` (internal bibliography coverage update for closest learned-eviction references; non-canonical)

## Notes on scientific status

- Many docs under `docs/pairwise_*` are intentionally exploratory theorem-development artifacts.
- Experimental policy docs are conservative by design and should not be read as finalized theorem claims.
