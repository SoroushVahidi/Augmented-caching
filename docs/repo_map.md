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

1. **[Reproducibility Guide](reproducibility.md)** (Environment, installation, workflow)
2. **[Canonical Experiments](canonical_experiments.md)** (Manuscript reproduction commands)
3. **[Dataset Setup](data_setup.md)** (Trace acquisition and preparation)
4. **[External Baselines](external_baselines.md)** (Baseline provenance and setup)
5. **[Artifact Policy](artifact_policy.md)** (Version control and retention rules)
6. `docs/kbs_manuscript_workflow.md` (Detailed narrative workflow)
7. `docs/evict_value_v1_kbs_canonical_artifacts.md` (heavy_r1-only inputs for KBS tables/figures)
8. `docs/results_guide.md` (Results interpretation and pitfalls)
9. `docs/manuscript_evidence_map.md`
10. `docs/baselines.md`

## Notes on scientific status

- Many docs under `docs/pairwise_*` are intentionally exploratory theorem-development artifacts.
- Experimental policy docs are conservative by design and should not be read as finalized theorem claims.
