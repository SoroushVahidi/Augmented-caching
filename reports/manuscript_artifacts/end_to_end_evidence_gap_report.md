# End-to-end replay evidence gap (`evict_value_v1` / Wulver `heavy_r1`)

**Scope:** Repository-only assessment (no external `main.tex`). **Goal:** Identify the smallest supported step that upgrades the **canonical** online trace-replay story for the main `evict_value_v1` line.

**Conservative rule:** Do not cite `analysis/evict_value_wulver_v1_policy_comparison.csv` (unsuffixed) or `*_heavy_smoke.*` as the KBS main quantitative comparison; see `docs/wulver_heavy_evict_value_experiment.md` and `CANONICAL_KBS_SUBMISSION.md`.

---

## 1. What is the manuscript’s main end-to-end evidence gap?

The **designated** multi-trace, multi-capacity **online replay** comparison that feeds the manuscript builder’s **Table~3** and **Figures~2–3** is missing its **single quantitative source file**:

- **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`** (companion **`…_heavy_r1.md`** optional but useful for prose).

Until that CSV exists:

- `scripts/paper/build_kbs_main_manuscript_artifacts.py` keeps **offline** evidence fresh (Fig.~1, Fig.~4, Table~4, etc.) but leaves **Table~3** as an explicit “unavailable” stub and **Fig.~2–3** as **comment-only** snippets (`reports/manuscript_artifacts/manuscript_artifact_report.md`, `manuscript_artifact_manifest.json` with `policy_comparison_present: false`).

So the strongest empirical limitation is **not** “more baselines” or “new benchmarks” in-repo—it is the **absence of the one canonical eval artifact** that the paper’s own workflow already names as the gate for the main replay table and figures.

---

## 2. What existing artifacts already help?

| Artifact / area | Role | Citable as **main** KBS Wulver replay? |
|-----------------|------|----------------------------------------|
| `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`, `…_best_config_heavy_r1.json`, `…_train_metrics_heavy_r1.json` | Offline model selection / ablation (Table~4 / Fig.~4; Table~5 / Fig.~5 when policy CSV absent) | **No** for end-to-end replay ranking (explicitly framed as offline in builder captions). |
| `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`, `data/derived/evict_value_v1_wulver_heavy_r1/` | Dataset and split provenance | **Support** for Methods / setup; not replay misses vs baselines. |
| `models/evict_value_wulver_v1_best_heavy_r1.pkl` | Trained model consumed by heavy eval | **Prerequisite** for canonical eval (present in this workspace as of report authoring). |
| `analysis/wulver_trace_manifest_full.csv` | Seven-family manifest used by eval driver | **Design** evidence; eval output still required for numbers. |
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_smoke.csv` | Tiny 2-trace, 2-capacity wiring replay | **No** — smoke / non-`heavy_r1` path per `analysis/README.md` and `reports/manuscript_artifacts/heavy_smoke_audit.md`. |
| `analysis/failure_slice_audit_heavy_r1_bounded.*` | Decision-slice diagnostics | **Niche**; current bounded audit shows model load issues and empty slices—**not** a substitute for full policy comparison. |
| Exploratory trees (e.g. `analysis/pairwise_*_campaign/`) | Other research threads | **No** for canonical KBS `heavy_r1` claims unless explicitly scoped in text. |

**Bottom line:** Offline training evidence is **already strong in-repo**; the **replay comparison row** for the same trained model vs the **documented baseline set** is what is missing.

---

## 3. What minimal additional experiment(s) would help most?

**One supported action** dominates improvement per unit effort:

1. **Run the existing heavy eval stage** (only), with defaults already wired to the canonical baseline list and manifest:

   ```bash
   sbatch --export=ALL,EXP_TAG=heavy_r1 slurm/evict_value_v1_wulver_heavy_eval.sbatch
   ```

   Preconditions (from `slurm/evict_value_v1_wulver_heavy_eval.sbatch` and `docs/wulver_heavy_evict_value_experiment.md`):

   - `analysis/wulver_trace_manifest_full.csv` exists.
   - `models/evict_value_wulver_v1_best_heavy_r1.pkl` exists (eval default `EVICT_VALUE_MODEL`).

2. **After the CSV appears**, refresh manuscript outputs:

   ```bash
   export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
   test -f analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv
   python scripts/paper/build_kbs_main_manuscript_artifacts.py
   ```

That pair reproduces **Table~3**, **Fig.~2–3**, tightens **Table~1** “Main” alignment to the eval roster (per builder logic), and **drops** offline-only Table~5 / Fig.~5 from the quantitative main line (builder replaces them with placeholders when policy CSV is present).

**Not recommended** as a substitute for the main table (without a new, explicitly documented tag and builder change):

- Renaming or copying `heavy_smoke` or unsuffixed policy CSVs to `…_heavy_r1.csv` — violates the repo’s **suffix = provenance** convention.
- Broad new sweeps (`pairwise_*`, extra policies in default comparison drivers) — outside “smallest set” for this paper track.

**Optional second-tier** (only if needed after eval completes):

- Re-run `build_kbs_main_manuscript_artifacts.py` whenever training or eval inputs change (manifest refresh, caption tweaks).
- If slice-level narrative is desired later, fix model loading for bounded audits and re-drive the relevant script **as a supplement**, not as replacement for `policy_comparison_heavy_r1.csv`.

---

## 4. Exact files, scripts, and commands

| Step | Path / command |
|------|----------------|
| Canonical runbook | `docs/wulver_heavy_evict_value_experiment.md` |
| Workflow hub | `docs/kbs_manuscript_workflow.md` |
| Artifact names | `docs/evict_value_v1_kbs_canonical_artifacts.md` |
| Eval driver | `slurm/evict_value_v1_wulver_heavy_eval.sbatch` |
| Underlying CLI | `python scripts/run_policy_comparison_wulver_v1.py` with the same arguments the sbatch file passes (`--trace-manifest`, `--capacities`, `--max-requests-per-trace`, `--policies`, `--evict-value-model`, `--out-csv`, `--out-md`) |
| Manuscript refresh | `python scripts/paper/build_kbs_main_manuscript_artifacts.py` |
| Builder inputs map | `EVIDENCE_FILES` in `scripts/paper/build_kbs_main_manuscript_artifacts.py` |
| Evidence status | `reports/manuscript_artifacts/manuscript_artifact_report.md`, `manuscript_artifact_manifest.json` |

**Resource note (from repo docs):** `docs/wulver_heavy_evict_value_experiment.md` documents that a **24h** eval **timed out** before writing the canonical CSV/MD; the eval sbatch requests **72:00:00**. Treat walltime reductions as **risky** unless re-measured.

---

## 5. Are “pending jobs” enough by themselves?

**Yes, if** the pending work is specifically **`heavy_r1` eval completion** (the job that writes `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` + `.md`). No additional **new** benchmark design is required in-repo: train-side `*_heavy_r1` artifacts and the tagged model file already match what the eval sbatch expects.

**No**, if “pending jobs” means anything **other** than that eval (e.g. more smoke runs, exploratory campaigns, or offline-only training reruns)—those do **not** unlock Table~3 / Fig.~2–3 in the canonical builder.

---

## 6. Manuscript snippets from *current* outputs (without policy CSV)

**Not possible** for the **canonical** Table~3 / Fig.~2–3 bundle: the builder is intentionally gated on `…_policy_comparison_heavy_r1.csv`. Generating “main” snippets from `heavy_smoke` or unsuffixed policy CSVs would **contradict** the repository’s KBS submission rules.

After the canonical CSV exists, run **`build_kbs_main_manuscript_artifacts.py`** once; it regenerates the relevant `tables/manuscript/*`, `figures/manuscript/figure{2,3}_*`, and `reports/manuscript_artifacts/latex_snippets/{table3,figure2,figure3}_snippet.tex` automatically.

---

## 7. Quick verification checklist (authors)

```bash
test -f models/evict_value_wulver_v1_best_heavy_r1.pkl
test -f analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv
test -f analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv   # should succeed after eval
```

Then rebuild manuscript artifacts and confirm `manuscript_artifact_manifest.json` has `"policy_comparison_present": true`.
