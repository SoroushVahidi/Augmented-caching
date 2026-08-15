# Canonical KBS submission path (`evict_value_v1` / Wulver `heavy_r1`)

**Single gateway document** for the **Knowledge-Based Systems** manuscript line: what to run, what to cite, and what to avoid. This file does not replace detailed runbooks; it points to them.

**Evidence strength:** The repository separates **designated canonical filenames** from legacy and exploratory outputs. Do not treat “any CSV under `analysis/`” as interchangeable with the paths below.

---

## 1. What this path is

- **Method:** `evict_value_v1` — learned candidate-level eviction scores trained on Wulver-scale artifact-backed datasets, evaluated by **trace replay** alongside documented baselines.
- **Experiment tag:** `EXP_TAG=heavy_r1` on Slurm drivers (train then eval).
- **Manuscript bundle:** `scripts/paper/build_kbs_main_manuscript_artifacts.py` reads **only** the canonical `*_heavy_r1` analysis inputs listed in `docs/evict_value_v1_kbs_canonical_artifacts.md` and writes `tables/manuscript/`, `figures/manuscript/`, `reports/manuscript_artifacts/`.

---

## 2. Exact scripts (canonical drivers)

| Stage | Script / file |
|-------|----------------|
| Train (dataset + model) | `slurm/evict_value_v1_wulver_heavy_train.sbatch` with `sbatch --export=ALL,EXP_TAG=heavy_r1,...` |
| Eval (policy replay comparison) | `slurm/evict_value_v1_wulver_heavy_eval.sbatch` with `sbatch --export=ALL,EXP_TAG=heavy_r1,...` |
| Manuscript tables/figures | `python scripts/paper/build_kbs_main_manuscript_artifacts.py` (from repo root; see `scripts/paper/README.md`) |

**Runbook (defaults, success checks, logs):** `docs/wulver_heavy_evict_value_experiment.md`

**Wiring-only smoke (not canonical KBS numbers):** `slurm/evict_value_v1_wulver_heavy_smoke.sbatch` — see `analysis/README.md`.

---

## 3. Exact canonical inputs

| Role | Path |
|------|------|
| Trace manifest | `analysis/wulver_trace_manifest_full.csv` |
| Dataset / train / model-selection outputs | Paths under `docs/evict_value_v1_kbs_canonical_artifacts.md` (`*_heavy_r1` filenames) |
| Trained model (eval expects tagged copy) | `models/evict_value_wulver_v1_best_heavy_r1.pkl` (also `models/evict_value_wulver_v1_best.pkl` during training) |

---

## 4. Exact canonical outputs (analysis)

**Authoritative list:** `docs/evict_value_v1_kbs_canonical_artifacts.md`

Minimum set for the **full** quantitative manuscript bundle (Table~3, Fig.~2–3):

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` (**required** for main replay table/figures)
- Optional companion: `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md`

**Also used by the builder** (offline ablation, dataset summary, etc.): same doc’s table of `*_heavy_r1` paths.

**Derived data on disk (train stage):** `data/derived/evict_value_v1_wulver_heavy_r1/` (`manifest.json`, `split_summary.csv`, …) per `docs/wulver_heavy_evict_value_experiment.md`.

---

## 5. Manuscript tables / figures (generated)

After a successful builder run:

- **`tables/manuscript/`** — `table1_*` … `table5_*` (see `reports/manuscript_artifacts/manuscript_artifact_report.md` for which tables are stubs when policy CSV is missing).
- **`figures/manuscript/`** — `figure1_*`, `figure4_*`, …; Fig.~2–3 require the policy comparison CSV.
- **`reports/manuscript_artifacts/`** — manifest, report, optional `latex_snippets/`.

**Preflight before claiming main numbers:**

```bash
test -f analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv
```

---

## 6. What **not** to cite for main KBS Wulver claims

| Do **not** use as primary KBS evidence | Why |
|----------------------------------------|-----|
| `analysis/evict_value_wulver_v1_policy_comparison.csv` (no `_heavy_r1`) | Often from broader drivers; may include extra policies (`atlas_v3`, `ml_gate_*`, …) per `docs/wulver_heavy_evict_value_experiment.md` |
| Unsuffixed `*_train_metrics.json`, `*_model_comparison.csv`, `*_best_config.json` at `analysis/` root | Parallel to older runs; canonical uses `*_heavy_r1` suffix |
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_smoke.*` | Smoke / wiring check only |
| `analysis/pairwise_*_campaign/` and large `jobs/` trees | Exploratory campaigns unless explicitly cross-walked |
| `analysis/*_light/` directories | Lightweight exploratory ablations (see `docs/lightweight_exploratory_ablations.md`) |
| `docs/pairwise_*` theorem sketches | Research-active; not the KBS quantitative path |

---

## 7. Exploratory-only (keep available, label clearly)

- **Pairwise / ranking line:** separate scripts and policies; see `docs/manuscript_evidence_map.md`, `docs/reproducibility_and_artifacts.md`.
- **Internal working docs:** `docs/internal_*` — author tooling, not canonical evidence.
- **Method rewrite support (internal):** `docs/method_detail_support_evict_value_v1.md` — consolidates repo facts for Methods sections; not a result artifact.

---

## 8. Related navigation

| Document | Role |
|----------|------|
| `docs/kbs_manuscript_workflow.md` | Detailed workflow + “not canonical” table |
| `docs/kbs_manuscript_submission_index.md` | Reviewer-facing index |
| `docs/reproducibility_and_artifacts.md` | CLI entry points, output roots |
| `docs/README.md` | Index of all `docs/` |
| `analysis/README.md` | Canonical vs exploratory under `analysis/` |
| `scripts/README.md` | Script families (paths stay flat for compatibility) |

**Caveats and open questions:** `docs/manuscript_open_questions.md`, `docs/manuscript_evidence_map.md`.
