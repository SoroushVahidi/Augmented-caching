# Scripts guide

This directory contains reproducible entry points for data preparation, model training, and experiment evaluation.

## Directory structure

- `scripts/setup/` — dataset ingestion, download, and preprocessing drivers.
- `scripts/maintenance/` — search, theorem-support, and aggregation utilities.
- `scripts/validation/` — revision readiness, status, and overhead benchmark runners.
- `scripts/experiments/canonical/` — **manuscript-safe (KBS):** canonical Wulver pipeline drivers and `paper/` artifact builders.
- `scripts/experiments/reviewer/` — scripts for reviewer-revision experiments (e.g., external baselines, fairness).
- `scripts/experiments/exploratory/` — exploratory lightweight ablations and first-check runners.
- `scripts/experiments/diagnostics/` — failure-slice audits and diagnostic diagnostic tools.

## Naming conventions

- `build_<target>.py`: deterministic data/table generation.
- `train_<target>.py`: model fitting and model-selection summaries.
- `run_<experiment>.py`: end-to-end experiment execution.
- `search_<topic>.py`: exhaustive/heuristic search utilities (typically theorem/counterexample support).

## Output conventions

Prefer writing to a dedicated analysis directory:

- `--output-dir analysis/<experiment_name>`
- include `summary.json` + `report.md` + one or more CSV tables.

## Canonical KBS Wulver workflow

For the canonical **`heavy_r1`** submission path, refer to:
- `scripts/experiments/canonical/paper/build_kbs_main_manuscript_artifacts.py`
- `docs/kbs_manuscript_workflow.md`
- `CANONICAL_KBS_SUBMISSION.md` (root)
