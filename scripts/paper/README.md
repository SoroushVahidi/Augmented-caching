# Paper / manuscript scripts

## KBS main bundle (canonical `heavy_r1` evidence)

| Script | Purpose | Manuscript-safe? |
|--------|---------|------------------|
| `build_kbs_main_manuscript_artifacts.py` | Reads **`analysis/*_heavy_r1.*`** inputs from the Wulver heavy train/eval pipeline (see `docs/wulver_heavy_evict_value_experiment.md`), writes `tables/manuscript/`, `figures/manuscript/`, `reports/manuscript_artifacts/`. | **Yes** — only when every path in `EVIDENCE_FILES` inside the script exists (including `evict_value_wulver_v1_policy_comparison_heavy_r1.csv`). |

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
