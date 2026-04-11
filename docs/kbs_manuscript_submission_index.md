# KBS manuscript: submission index (`evict_value_v1` / Wulver)

Single entry point for reviewers and authors: what is **canonical** for the main Wulver learned-eviction story, what is **exploratory**, and where tables/figures come from.

## Canonical evidence path (use for main quantitative claims)

| Step | Reference |
|------|-----------|
| Runbook | `docs/wulver_heavy_evict_value_experiment.md` |
| Train (dataset + model) | `slurm/evict_value_v1_wulver_heavy_train.sbatch` with `EXP_TAG=heavy_r1` |
| Eval (policy comparison) | `slurm/evict_value_v1_wulver_heavy_eval.sbatch` with `EXP_TAG=heavy_r1` |
| Input/output filenames | `docs/evict_value_v1_kbs_canonical_artifacts.md` |
| Tables / figures / manifest | `scripts/paper/build_kbs_main_manuscript_artifacts.py` → `tables/manuscript/`, `figures/manuscript/`, `reports/manuscript_artifacts/` |

The Python entry points invoked by those batch files are the **only** drivers designated for the **`_heavy_r1`** analysis artifacts used by `EVIDENCE_FILES` in `build_kbs_main_manuscript_artifacts.py`.

## Non-canonical but retained (do not cite as main KBS Wulver comparison)

Files may remain under `analysis/` for history and reproducibility of older or broader runs:

| Path pattern | Role |
|--------------|------|
| `analysis/evict_value_wulver_v1_policy_comparison.csv` / `.md` | Default output name from `run_policy_comparison_wulver_v1.py`; often includes **extra policies** (e.g. `ml_gate_*`, `atlas_v3`) when not using `--policies` as in heavy eval. **Not** the KBS primary comparison. |
| `analysis/evict_value_wulver_v1_{train_metrics,model_comparison,best_config}.json` / `.csv` (no `_heavy_r1`) | Parallel outputs from non-tagged or alternate training runs. |
| `analysis/evict_value_v1_wulver_dataset_summary.md` (no `_heavy_r1`) | Older summary; canonical summary for KBS is `..._dataset_summary_heavy_r1.md`. |

## Exploratory (not the main `heavy_r1` line)

- **Slurm:** `evict_value_v1_wulver_multi_phase.sbatch`, `evict_value_v1_wulver_dataset.sbatch`, `evict_value_v1_wulver_dataset_array.sbatch` — see comments in each file; useful for scaling tests or dataset-only builds, not interchangeable with the KBS canonical filenames without explicit verification.
- **Docs / analysis:** `docs/pairwise_*`, `analysis/pairwise_*_campaign/` — theorem development and campaign outputs; interpret per `docs/manuscript_evidence_map.md`.
- **Lightweight ablations:** incoming-aware/history-aware/history-objective/joint-state outputs under `analysis/*_light/`; indexed in `docs/lightweight_exploratory_ablations.md`.

## Once heavy eval finishes: handoff commands

```bash
test -f analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv
python scripts/paper/build_kbs_main_manuscript_artifacts.py
```

If the first command fails, the heavy eval stage is not complete yet.

## Open dependency (repository hygiene only)

Final committed **`tables/manuscript/`** and **`figures/manuscript/`** are **only** guaranteed to match canonical **`_heavy_r1`** policy comparison inputs **after** `build_kbs_main_manuscript_artifacts.py` has been run successfully with **all** `EVIDENCE_FILES` present (including `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`). If that file is missing, the builder exits with an error; do not infer completion from older figure timestamps alone.
