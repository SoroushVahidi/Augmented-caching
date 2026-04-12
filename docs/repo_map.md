# Repository map

This document is a concise orientation guide for external readers and manuscript reviewers.

## KBS manuscript workflow (submission path)

For **Knowledge-Based Systems** and the canonical Wulver **`heavy_r1`** `evict_value_v1` line (Slurm drivers, evidence files, `build_kbs_main_manuscript_artifacts.py`, `tables/manuscript/`, `figures/manuscript/`), use **`docs/kbs_manuscript_workflow.md`** as the single entry point. Reviewer index: `docs/kbs_manuscript_submission_index.md`.

## Top-level layout

- `src/lafc/` — core library implementation (policies, simulator, runners, datasets, offline solvers).
- `scripts/` — reproducible experiment and dataset-prep entry points.
- `tests/` — unit/integration tests for policies, datasets, runners, and experiments.
- `docs/` — method notes, experiment protocols, theorem-development notes, and manuscript-support docs.
- `analysis/` — generated text artifacts from experiments (`.csv`, `.json`, `.md`).
- `data/` — small examples in git + raw/processed derived data roots.
- `slurm/` — cluster batch templates for heavier runs. **KBS canonical Wulver pipeline:** `evict_value_v1_wulver_heavy_train.sbatch`, `evict_value_v1_wulver_heavy_eval.sbatch` (`EXP_TAG=heavy_r1`); index at `docs/kbs_manuscript_submission_index.md`.

## `src/lafc/` subpackages

- `policies/` — policy implementations (baseline, robust, and experimental).
- `simulator/` — cache state and request trace execution logic.
- `runner/` — CLI entrypoint (`python -m lafc.runner.run_policy`).
- `datasets/` — dataset ingestion, preprocessing, and CLI glue.
- `offline/` — offline reference solvers and IO/helpers.
- `learned_gate/` — v1/v2 learned gate datasets, features, and models.
- top-level `evict_*` / `offline_teacher_supervision.py` — eviction-learning datasets/models and supervision helpers.
- `metrics/` — common cost and prediction error metrics.

## `scripts/` families

- `scripts/paper/` — KBS manuscript bundle (`build_kbs_main_manuscript_artifacts.py`); requires canonical `heavy_r1` analysis inputs.
- `scripts/datasets/` — canonical dataset download/prepare wrappers.
- `build_*` scripts — build training tables from traces.
- `train_*` scripts — fit lightweight models and write metrics.
- `run_*` scripts — first-checks, ablations, and experiment reports.
- `search_*` / `analyze_*` scripts — theorem/proof or counterexample support utilities.
- `scripts/experiments/` — lightweight ablation runners (incoming-aware, history-aware, history-objective, joint-state).

## `analysis/` organization conventions

- **Experiment directories** (preferred): one directory per experiment with `report.md`, `results.csv`, `summary.json`.
- **Legacy root-level artifacts**: older single-file outputs kept for history and manuscript traceability.
- **Manifests/audits**: stable helper artifacts (for example Wulver trace manifests and failure-slice audits).
- **Exploratory lightweight ablations**: grouped under `analysis/*_light/` (see `docs/lightweight_exploratory_ablations.md`).

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

## Notes on scientific status

- Many docs under `docs/pairwise_*` are intentionally exploratory theorem-development artifacts.
- Experimental policy docs are conservative by design and should not be read as finalized theorem claims.
