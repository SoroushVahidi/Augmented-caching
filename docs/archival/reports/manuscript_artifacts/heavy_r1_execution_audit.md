# heavy_r1 execution audit — KBS canonical manuscript path

**Date (UTC):** 2026-04-12  
**Repository:** `Augmented-caching` on Wulver  
**Goal:** Canonical `evict_value_v1` `heavy_r1` evidence → `build_kbs_main_manuscript_artifacts.py` → manuscript tables/figures.

---

## 1. What existed before this run

Commands:

```bash
cd /project/ikoutis/sv96/Augmented-caching
for f in \
  analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv \
  analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md \
  analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md \
  analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json \
  analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv \
  analysis/evict_value_wulver_v1_best_config_heavy_r1.json \
  analysis/wulver_trace_manifest_full.csv \
  models/evict_value_wulver_v1_best_heavy_r1.pkl \
  data/derived/evict_value_v1_wulver_heavy_r1/manifest.json; do
  if [ -f "$f" ]; then wc -c "$f"; else echo "MISSING $f"; fi
done
```

**Result:**

| Canonical artifact | Present | Notes |
|--------------------|---------|--------|
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` | **No** | Required for Table~3 / Fig.~2–3 |
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md` | **No** | Optional companion report |
| `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` | Yes | Non-empty |
| `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json` | Yes | Non-empty |
| `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` | Yes | Non-empty |
| `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` | Yes | Non-empty |
| `analysis/wulver_trace_manifest_full.csv` | Yes | Non-empty |
| `models/evict_value_wulver_v1_best_heavy_r1.pkl` | Yes | Non-empty |
| `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` | Yes | Non-empty |

**Manuscript outputs already on disk:** `tables/manuscript/`, `figures/manuscript/`, `reports/manuscript_artifacts/` were populated from earlier builder runs **without** the policy comparison CSV (no `figure2_*` / `figure3_*`; Table~3 stub; offline Table~5 / Fig.~5 path).

---

## 2. Jobs submitted

Canonical train batch **was not** submitted: training-stage artifacts and `models/evict_value_wulver_v1_best_heavy_r1.pkl` already exist per `docs/wulver_heavy_evict_value_experiment.md`, so re-running `slurm/evict_value_v1_wulver_heavy_train.sbatch` would duplicate ~36h work and overwrite tagged outputs unless `BUILD_OVERWRITE` is used.

Only the **eval** stage was submitted, matching `docs/wulver_heavy_evict_value_experiment.md` (eval writes `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.{csv,md}`).

Command:

```bash
cd /project/ikoutis/sv96/Augmented-caching
sbatch --parsable --export=ALL,EXP_TAG=heavy_r1 slurm/evict_value_v1_wulver_heavy_eval.sbatch
```

| Job ID | Script | EXP_TAG | Purpose |
|--------|--------|---------|---------|
| **910352** | `slurm/evict_value_v1_wulver_heavy_eval.sbatch` | `heavy_r1` | Policy comparison on full manifest |

---

## 3. Job outcomes (as of audit write)

```bash
squeue -j 910352
sacct -j 910352 --format=JobID,JobName,State,ExitCode,Elapsed,Reason -n -P
```

**State:** **PENDING** — reason `(ReqNodeNotAvail, Reserved for maintenance)` on partition `general`. **Not started**; no Slurm stdout/stderr beyond queue state yet.

**Success:** **Unknown / not completed** — cannot verify `ExitCode` or presence of canonical CSV/MD until the job runs and finishes.

---

## 4. Post-completion checklist (run after 910352 is **COMPLETED**)

```bash
cd /project/ikoutis/sv96/Augmented-caching
test -s analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv
test -s analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
python scripts/paper/build_kbs_main_manuscript_artifacts.py
grep -q '"policy_comparison_present": true' reports/manuscript_artifacts/manuscript_artifact_manifest.json
ls tables/manuscript/table3_main_quantitative_comparison.tex figures/manuscript/figure2_main_performance_comparison.pdf figures/manuscript/figure3_improvement_vs_lru.pdf
```

Expected logs after success: `slurm/logs/evictv1-heavy-eval-910352.out` and `.err`.

---

## 5. Builder run in this session

**Not executed** after `heavy_r1` eval, because the canonical policy comparison files do not exist yet. Running the builder now would only reproduce the **partial** bundle (same as prior commits: no Fig.~2/3, stub Table~3).

---

## 6. Documentation conflict (noted for maintainers)

- **`docs/evict_value_v1_kbs_canonical_artifacts.md`** states the manuscript builder fails until `policy_comparison_heavy_r1.csv` exists.
- **Actual behavior** (`scripts/paper/build_kbs_main_manuscript_artifacts.py`): if **core** `_heavy_r1` training/dataset files exist but the policy CSV is missing, `main()` **exits successfully** and emits stubs / offline supplements (see `manuscript_artifact_report.md`).

For this audit, the **manuscript-specific** definition of “ready” follows `docs/kbs_manuscript_workflow.md` / `scripts/paper/README.md`: full main quantitative bundle requires **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`**.

---

## 7. Exact canonical files (target state after successful eval + build)

Per `docs/evict_value_v1_kbs_canonical_artifacts.md` and `EVIDENCE_FILES` in `build_kbs_main_manuscript_artifacts.py`:

**Analysis inputs**

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md` (optional note in builder if CSV-only)
- `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`
- `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json`
- `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`
- `analysis/evict_value_wulver_v1_best_config_heavy_r1.json`
- `analysis/wulver_trace_manifest_full.csv`

**Generated manuscript roots**

- `tables/manuscript/`
- `figures/manuscript/`
- `reports/manuscript_artifacts/` (incl. `manuscript_artifact_manifest.json`, `manuscript_artifact_report.md`, `latex_snippets/`)

---

## 8. Deviations / missing items

- **Missing:** `policy_comparison_heavy_r1.{csv,md}` — blocking main KBS quantitative tables/figures.
- **Eval job** not yet running due to cluster maintenance reservation.
- **Train job** skipped by design given existing `heavy_r1` train artifacts (documented above).

---

## 9. Verdict (initial audit)

**NOT READY** — as of the first audit write.

---

## 10. Resume session — 2026-04-12 (monitor canonical eval 910352)

### Commands run

```bash
cd /project/ikoutis/sv96/Augmented-caching
squeue -j 910352
sacct -j 910352 --format=JobID,JobName,State,ExitCode,Elapsed,Reason,End -n -P
test -f analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv && wc -c analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv
```

### Job 910352 outcome (this session)

| Field | Value |
|--------|--------|
| **State** | `PENDING` — job **not started** (no allocation yet) |
| **Reason** | `(ReqNodeNotAvail, Reserved for maintenance)` |
| **Elapsed** | `00:00:00` on main job while pending |
| **Partition / QoS** | `general` / `standard` (as in `slurm/evict_value_v1_wulver_heavy_eval.sbatch`; unchanged) |

**Slurm logs:** `slurm/logs/evictv1-heavy-eval-910352.out` and `.err` **do not exist** yet — expected until Slurm allocates a node and the batch script runs.

### Canonical eval outputs (verified absent)

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` — **missing**
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md` — **missing**

### Training / model precondition

Not rerun per constraints: existing `heavy_r1` training artifacts, `models/evict_value_wulver_v1_best_heavy_r1.pkl`, and `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` remain the intended inputs for eval once job **910352** starts.

### Manuscript artifact builder

**Not executed** in this session: without `policy_comparison_heavy_r1.csv`, a builder run would **not** produce manuscript-verified Table~3 / Fig.~2–3 from end-to-end replay evidence (see `docs/kbs_manuscript_workflow.md`). Prior partial outputs under `tables/manuscript/` / `figures/manuscript/` may still reflect the stub path (`policy_comparison_present: false`).

### Corrective action

- **None** that resubmits work: while **910352** remains a valid pending job for user `sv96`, **did not** submit a second `evict_value_v1_wulver_heavy_eval.sbatch` — duplicate jobs could waste resources and contend for the same canonical filenames.
- **Did not** rerun `slurm/evict_value_v1_wulver_heavy_train.sbatch`.

### Doc conflict (unchanged)

`docs/evict_value_v1_kbs_canonical_artifacts.md` implies the builder “fails” without the policy CSV; the script **exits 0** with stubs if core train files exist. For **manuscript readiness**, the stricter rule in `docs/kbs_manuscript_workflow.md` applies: full quantitative bundle needs the policy CSV.

### Verdict (resume)

**Canonical path incomplete:** cluster maintenance blocks **910352** from starting; `heavy_r1` policy comparison files are still absent. When **910352** finishes with `State=COMPLETED` and `ExitCode=0:0`, run §4 checklist, then `python scripts/paper/build_kbs_main_manuscript_artifacts.py`.

**If 910352 is cancelled, fails, or ages out:** resubmit **only** `sbatch --export=ALL,EXP_TAG=heavy_r1 slurm/evict_value_v1_wulver_heavy_eval.sbatch` after confirming the tagged model still exists; do not use unsuffixed `analysis/evict_value_wulver_v1_policy_comparison.csv` for KBS main claims.
