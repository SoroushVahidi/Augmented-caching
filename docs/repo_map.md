# Repository map

This document is a concise orientation guide for external readers and manuscript reviewers.

## Top-level layout

- `src/lafc/` — core library implementation (policies, simulator, runners, datasets, offline solvers).
- `scripts/` — reproducible experiment and dataset-prep entry points.
- `tests/` — unit/integration tests for policies, datasets, runners, and experiments.
- `docs/` — method notes, experiment protocols, theorem-development notes, and manuscript-support docs.
- `analysis/` — generated text artifacts from experiments (`.csv`, `.json`, `.md`).
- `data/` — small examples in git + raw/processed derived data roots.
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

## `scripts/` families

- `scripts/datasets/` — canonical dataset download/prepare wrappers.
- `build_*` scripts — build training tables from traces.
- `train_*` scripts — fit lightweight models and write metrics.
- `run_*` scripts — first-checks, ablations, and experiment reports.
- `search_*` / `analyze_*` scripts — theorem/proof or counterexample support utilities.

## `analysis/` organization conventions

- **Experiment directories** (preferred): one directory per experiment with `report.md`, `results.csv`, `summary.json`.
- **Legacy root-level artifacts**: older single-file outputs kept for history and manuscript traceability.
- **Manifests/audits**: stable helper artifacts (for example Wulver trace manifests and failure-slice audits).

See `analysis/README.md` for details and naming guidance.

## Manuscript-support docs to read first

1. `docs/reproducibility_and_artifacts.md` (entry points, output locations, manuscript vs exploratory)
2. `docs/manuscript_evidence_map.md`
3. `docs/manuscript_open_questions.md`
4. `docs/baselines.md`
5. `docs/framework.md`

## Notes on scientific status

- Many docs under `docs/pairwise_*` are intentionally exploratory theorem-development artifacts.
- Experimental policy docs are conservative by design and should not be read as finalized theorem claims.
