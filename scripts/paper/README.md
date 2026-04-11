# Paper / manuscript scripts

**Primary doc:** `docs/kbs_manuscript_workflow.md` — canonical `heavy_r1` inputs, this builder, and output directories (`tables/manuscript/`, `figures/manuscript/`, `reports/manuscript_artifacts/`).

## KBS main bundle (canonical `heavy_r1` evidence)

| Script | Purpose | Manuscript-safe? |
|--------|---------|------------------|
| `build_kbs_main_manuscript_artifacts.py` | Reads canonical **`_heavy_r1`** artifacts; always refreshes Fig.~1, Fig.~4, Table~2, Table~4. If `evict_value_wulver_v1_policy_comparison_heavy_r1.csv` is present, also rebuilds Table~1, Table~3, Fig.~2 (`figure2_main_performance_comparison`), Fig.~3 (`figure3_improvement_vs_lru`); otherwise writes **explicit unavailable stubs** for policy-dependent outputs (see `manuscript_artifact_report.md`). | **Yes** for refreshed items; policy stubs are **not** citable as main results. |
| `regenerate_evidence_aligned_manuscript_figures.py` | Regenerates **Figure 1** (method schematic) and **Figure 4** (offline training ablation) as vector PDF + PNG using `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`. Does **not** require policy-comparison CSV. | **Yes** for those figures only. |
| `build_kbs_manuscript_pre_eval_artifacts.py` | Tables + same Figure 1/4 assets as above, plus dataset `longtable` inputs, without reading policy comparison. | **Yes** (offline / schematic evidence only). |

**Not manuscript-primary:** other tooling under `scripts/` that emits `analysis/` for pairwise campaigns, theorem checks, or ad hoc comparisons — see `scripts/README.md` (“Exploratory and non-canonical drivers”).

Run from repository root:

```bash
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
python scripts/paper/build_kbs_main_manuscript_artifacts.py
```

Quick preflight:

```bash
test -f analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv
```

If preflight fails, finish the heavy eval job first (`slurm/evict_value_v1_wulver_heavy_eval.sbatch` with `EXP_TAG=heavy_r1`).

## Figures without online eval (method + offline ablation only)

```bash
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
python scripts/paper/regenerate_evidence_aligned_manuscript_figures.py
# or full pre-eval bundle (tables + figures 1 and 4):
python scripts/paper/build_kbs_manuscript_pre_eval_artifacts.py
```
