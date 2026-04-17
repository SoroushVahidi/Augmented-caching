# Scripts guide

This directory contains reproducible entry points for data preparation, model training, and experiment evaluation.

## Directory structure

- `scripts/datasets/` — dataset ingestion/preprocessing drivers.
- root-level `build_*` — construct training/evaluation tables.
- root-level `train_*` — train lightweight models and export metrics.
- root-level `run_*` — execute experiments and write analysis artifacts.
- root-level `search_*` / `analyze_*` — targeted analysis/proof-support tooling.

## Naming conventions

- `build_<target>.py`: deterministic data/table generation.
- `train_<target>.py`: model fitting and model-selection summaries.
- `run_<experiment>.py`: end-to-end experiment with outputs under `analysis/<experiment>/` when possible.
- `search_<topic>.py`: exhaustive/heuristic search utilities (typically theorem/counterexample support).

## Output conventions

Prefer writing to a dedicated analysis directory:

- `--output-dir analysis/<experiment_name>`
- include `summary.json` + `report.md` + one or more CSV tables.

Legacy scripts that write root-level artifacts in `analysis/` are kept for backward compatibility.
