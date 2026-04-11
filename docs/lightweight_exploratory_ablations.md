# Lightweight exploratory ablations (non-canonical for KBS main manuscript)

This index makes lightweight branches easy to find **without** confusing them with the canonical Wulver `heavy_r1` manuscript path.

## Canonical boundary

The **only** main-paper Wulver `evict_value_v1` path is:

1. `slurm/evict_value_v1_wulver_heavy_train.sbatch` (`EXP_TAG=heavy_r1`)
2. `slurm/evict_value_v1_wulver_heavy_eval.sbatch` (`EXP_TAG=heavy_r1`)
3. `analysis/*_heavy_r1.*` evidence listed in `docs/evict_value_v1_kbs_canonical_artifacts.md`
4. `scripts/paper/build_kbs_main_manuscript_artifacts.py`

Everything below is exploratory/supporting and must not be cited as the canonical KBS Wulver comparison unless explicitly cross-referenced in manuscript docs.

## Lightweight ablation families (exploratory)

### 1) Incoming-aware ablation

- Script: `scripts/experiments/run_incoming_file_aware_ablation.py`
- Output root: `analysis/incoming_file_aware_ablation_light/`
- Typical artifacts: `model_comparison.csv`, `downstream_replay.csv`, `summary.json`, `report.md`

### 2) History-aware ablation

- Script: `scripts/experiments/run_history_context_ablation.py`
- Output root: `analysis/history_context_ablation_light/`
- Typical artifacts: `model_comparison.csv`, `downstream_replay.csv`, `summary.json`, `report.md`

### 3) History-objective / history-pairwise-style objective ablation

- Script: `scripts/experiments/run_history_objective_ablation.py`
- Output root: `analysis/history_objective_ablation_light/`
- Typical artifacts: `model_comparison.csv`, `downstream_replay.csv`, `summary.json`, `report.md`

### 4) Joint-state ablations

- Dataset builder: `scripts/experiments/build_joint_cache_state_dataset.py`
- Joint cache-state modeling: `scripts/experiments/run_joint_cache_state_lightweight_eval.py` → `analysis/joint_cache_state_model_light/`
- Joint-state reasoning ablation: `scripts/experiments/run_joint_state_reasoning_ablation.py` → `analysis/joint_state_reasoning_light/`

## Related exploratory branches retained for context

- Pairwise publishability campaign: `analysis/pairwise_publishability_campaign/`
- Pairwise chain witness campaign: `analysis/pairwise_chain_witness_campaign/`
- Pairwise/pointwise focused experiments: `analysis/pairwise_vs_pointwise/`, `analysis/evict_value_pairwise_multitrace_eval/`

These are historically useful and manuscript-supporting in parts, but they are not interchangeable with canonical `heavy_r1` evidence files.

## Stale or superseded naming patterns

- Unsuffixed root files such as `analysis/evict_value_wulver_v1_policy_comparison.csv` are retained for provenance and older drivers.
- For the KBS main line, always prefer `_heavy_r1` suffixed files when a pair exists.
