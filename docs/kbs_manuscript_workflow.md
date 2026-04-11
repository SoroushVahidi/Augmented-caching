# KBS manuscript workflow (`evict_value_v1` / Wulver `heavy_r1`)

**Single navigation page** for finalized **Knowledge-Based Systems** submission work: canonical evidence, builder command, LaTeX-oriented outputs, and where exploratory material lives (so it is not mistaken for the main paper line).

---

## 1. Canonical path (main quantitative Wulver story)

Use **only** this pipeline for primary **multi-trace Wulver** `evict_value_v1` claims:

| Step | What to read or run |
|------|---------------------|
| Runbook | `docs/wulver_heavy_evict_value_experiment.md` |
| Train (Slurm) | `slurm/evict_value_v1_wulver_heavy_train.sbatch` with `EXP_TAG=heavy_r1` |
| Eval (Slurm) | `slurm/evict_value_v1_wulver_heavy_eval.sbatch` with `EXP_TAG=heavy_r1` |
| Exact filenames | `docs/evict_value_v1_kbs_canonical_artifacts.md` |
| Reviewer index | `docs/kbs_manuscript_submission_index.md` |

**Preflight:** the manuscript builder requires **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`**. If it is missing, the canonical eval stage is incomplete—finish or rerun heavy eval per the runbook (do not substitute unsuffixed `analysis/evict_value_wulver_v1_policy_comparison.csv`).

---

## 2. Build tables, figures, and submission manifest (after heavy eval)

From repository root:

```bash
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
test -f analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv   # must succeed
python scripts/paper/build_kbs_main_manuscript_artifacts.py
```

**Generated outputs** (from `build_kbs_main_manuscript_artifacts.py`):

| Output area | Role |
|-------------|------|
| `tables/manuscript/` | CSV + `.tex` fragments for tables |
| `figures/manuscript/` | PDF + PNG figures |
| `reports/manuscript_artifacts/` | `manuscript_artifact_manifest.json`, `manuscript_artifact_report.md`, optional `latex_snippets/` |

Details: `scripts/paper/README.md`, `reports/manuscript_artifacts/` when present.

---

## 3. What is *not* canonical (exploratory / support)

These are **incoming-aware**, **history-aware**, **history-pairwise-style objectives**, **joint-state** runs, **pairwise campaigns**, **theorem/proof tooling**, and **multi-phase Wulver** drivers—use only with explicit caveats; they do **not** produce the designated `*_heavy_r1` filenames unless you deliberately retag outputs (not recommended for KBS).

| Bucket | Pointer |
|--------|---------|
| Lightweight ablations (`analysis/*_light/`) | `docs/lightweight_exploratory_ablations.md` |
| Pairwise / publishability / chain-witness campaigns | `analysis/pairwise_*_campaign/`, `docs/wulver_pairwise_*.md`, `docs/manuscript_evidence_map.md` |
| Non-heavy Slurm drivers (extra policies or dataset-only) | `slurm/evict_value_v1_wulver_multi_phase.sbatch`, `slurm/evict_value_v1_wulver_dataset*.sbatch` — see `scripts/README.md` |
| Unsuffixed `analysis/evict_value_wulver_v1_*.csv` / `.json` | Legacy or alternate runs; **not** interchangeable with `*_heavy_r1` for KBS main tables—see `analysis/README.md` |

---

## 4. Easy-to-confuse filenames (do not cite for main KBS Wulver line)

| Do **not** treat as main KBS comparison | Use instead |
|----------------------------------------|-------------|
| `analysis/evict_value_wulver_v1_policy_comparison.csv` | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` |
| `analysis/evict_value_wulver_v1_train_metrics.json` (no tag) | `..._train_metrics_heavy_r1.json` |
| `analysis/evict_value_v1_wulver_dataset_summary.md` (no tag) | `..._dataset_summary_heavy_r1.md` |

Full table: `analysis/README.md`, `docs/evict_value_v1_kbs_canonical_artifacts.md`.

---

## 5. Related reading (interpretation and scope)

- Evidence strength and claims: `docs/manuscript_evidence_map.md`
- Open questions: `docs/manuscript_open_questions.md`
- Reproducibility and CLI: `docs/reproducibility_and_artifacts.md`
- Repository layout: `docs/repo_map.md`
