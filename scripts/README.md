# Scripts guide

This directory contains reproducible entry points for data preparation, model training, and experiment evaluation.

## Directory structure

- `scripts/datasets/` — dataset ingestion/preprocessing drivers.
- `scripts/paper/` — **manuscript-safe (KBS):** `build_kbs_main_manuscript_artifacts.py` only when canonical `heavy_r1` inputs exist (`scripts/paper/README.md`).
- root-level `build_*` — construct training/evaluation tables.
- root-level `train_*` — train lightweight models and export metrics.
- root-level `run_*` — execute experiments and write analysis artifacts.
- `scripts/experiments/` — exploratory lightweight ablations (incoming-aware/history-aware/history-objective/joint-state).
- `run_offline_general_caching_approx.py` — offline LP+rounding baseline for general caching (not `python -m lafc.runner.run_policy`).
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

## Exploratory and non-canonical drivers (not the KBS `heavy_r1` main line)

These remain supported for research and scaling experiments; they are **not** the designated source of the **`_heavy_r1`** filenames used by `build_kbs_main_manuscript_artifacts.py`:

- **`slurm/evict_value_v1_wulver_multi_phase.sbatch`** — full pipeline with default `analysis/evict_value_wulver_v1_policy_comparison.csv` naming and optional `ml_gate` training; see file header.
- **`slurm/evict_value_v1_wulver_dataset.sbatch`** / **`evict_value_v1_wulver_dataset_array.sbatch`** — dataset generation only.
- **Pairwise / chain-witness / publishability campaigns:** `scripts/run_pairwise_*.py`, `scripts/aggregate_pairwise_*.py`, `slurm/pairwise_*_campaign_*.sbatch` — empirical and theorem-support outputs under `analysis/pairwise_*_campaign/`; exploratory for the main `evict_value_v1` Wulver story unless explicitly cross-referenced in `docs/manuscript_evidence_map.md`.

**Canonical KBS Wulver workflow:** `docs/kbs_manuscript_workflow.md` (also `docs/kbs_manuscript_submission_index.md`).

## Lightweight exploratory ablations (easy-to-find index)

These scripts and outputs are intentionally grouped as exploratory:

| Family | Script | Default output |
|---|---|---|
| Incoming-aware | `scripts/experiments/run_incoming_file_aware_ablation.py` | `analysis/incoming_file_aware_ablation_light/` |
| History-aware | `scripts/experiments/run_history_context_ablation.py` | `analysis/history_context_ablation_light/` |
| History-objective | `scripts/experiments/run_history_objective_ablation.py` | `analysis/history_objective_ablation_light/` |
| Joint cache-state model | `scripts/experiments/run_joint_cache_state_lightweight_eval.py` | `analysis/joint_cache_state_model_light/` |
| Joint-state reasoning | `scripts/experiments/run_joint_state_reasoning_ablation.py` | `analysis/joint_state_reasoning_light/` |

Canonical/non-canonical boundary doc: `docs/lightweight_exploratory_ablations.md`.
