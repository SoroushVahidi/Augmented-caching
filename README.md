# Learning-Augmented Caching (`lafc`)

Research code for **learning-augmented caching**: literature-faithful baselines, robust combiners, dataset pipelines for public and manifest-based traces, and reproducible experiment scripts (CSV / JSON / Markdown artifacts).

---

## What this repository is

- **Simulator and policies:** `python -m lafc.runner.run_policy` (`src/lafc/`) for trace replay of named policies.
- **Learned eviction (main line):** **`evict_value_v1`** — scores each cache resident (candidate) with a supervised model; evicts a minimizer. Optional **`evict_value_v1_guarded`** wrapper (`docs/guarded_robust_wrapper.md`).
- **Exploratory research:** pairwise / ranking supervision, sentinel variants, and theorem notes under `docs/pairwise_*` and `analysis/pairwise_*` — **not** the canonical KBS Wulver table path unless explicitly cross-walked.

**Evidence caveats:** `docs/manuscript_evidence_map.md`, `docs/manuscript_open_questions.md`.

---

## Stable baselines (manuscript-safe references)

Classical and robust policies exposed on the main CLI include `lru`, `marker`, `predictive_marker`, `trust_and_doubt`, `robust_ftp_d_marker` (`robust_ftp`), `blind_oracle_lru_combiner`, and the unweighted optimum `offline_belady` (full lookahead via trace construction). **General caching** (variable sizes/costs) uses a separate LP+rounding entry point: `scripts/run_offline_general_caching_approx.py` — see `docs/offline_general_caching_approx.md` (not a `--policy` on `run_policy`).

Full roster and literature pointers: `docs/baselines.md` (summary lists above).

---

## Canonical KBS manuscript path (Wulver `heavy_r1`)

For **Knowledge-Based Systems** and the **only** designated multi-trace Wulver `evict_value_v1` evidence line:

| Step | Resource |
|------|----------|
| **One-page checklist** | **[`CANONICAL_KBS_SUBMISSION.md`](CANONICAL_KBS_SUBMISSION.md)** (repo root) |
| Narrative workflow | [`docs/kbs_manuscript_workflow.md`](docs/kbs_manuscript_workflow.md) |
| Slurm train → eval | [`slurm/evict_value_v1_wulver_heavy_train.sbatch`](slurm/evict_value_v1_wulver_heavy_train.sbatch), [`slurm/evict_value_v1_wulver_heavy_eval.sbatch`](slurm/evict_value_v1_wulver_heavy_eval.sbatch) with `EXP_TAG=heavy_r1` |
| Runbook | [`docs/wulver_heavy_evict_value_experiment.md`](docs/wulver_heavy_evict_value_experiment.md) |
| Exact filenames | [`docs/evict_value_v1_kbs_canonical_artifacts.md`](docs/evict_value_v1_kbs_canonical_artifacts.md) |
| Tables / figures | `python scripts/paper/build_kbs_main_manuscript_artifacts.py` → `tables/manuscript/`, `figures/manuscript/`, `reports/manuscript_artifacts/` |

**Do not** cite `analysis/evict_value_wulver_v1_policy_comparison.csv` (unsuffixed) as the main KBS comparison; it may include extra policies from non-heavy drivers. Use **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`** only when present. See [`analysis/README.md`](analysis/README.md).

### After heavy eval completes (minimal checklist)

```bash
test -f analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
python scripts/paper/build_kbs_main_manuscript_artifacts.py
ls tables/manuscript figures/manuscript reports/manuscript_artifacts
```

If the first command fails, finish the eval stage per [`docs/wulver_heavy_evict_value_experiment.md`](docs/wulver_heavy_evict_value_experiment.md).

---

## Reproduce main artifacts (orientation)

| Goal | Start here |
|------|------------|
| KBS `heavy_r1` bundle | [`CANONICAL_KBS_SUBMISSION.md`](CANONICAL_KBS_SUBMISSION.md) |
| All docs (index) | [`docs/README.md`](docs/README.md) |
| CLI and output roots | [`docs/reproducibility_and_artifacts.md`](docs/reproducibility_and_artifacts.md) |
| `analysis/` layout | [`analysis/README.md`](analysis/README.md) |
| `scripts/` layout | [`scripts/README.md`](scripts/README.md) |
| Repo layout | [`docs/repo_map.md`](docs/repo_map.md) |

**Method detail (internal support for rewriting Methods):** [`docs/method_detail_support_evict_value_v1.md`](docs/method_detail_support_evict_value_v1.md) (consolidates `docs/evict_value_v1_method_spec.md` and related sources; not a result artifact).

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Quick start

### Baseline

```bash
python -m lafc.runner.run_policy \
  --policy predictive_marker \
  --trace data/example_unweighted.json \
  --capacity 3
```

### Robust combiner

```bash
python -m lafc.runner.run_policy \
  --policy robust_ftp_d_marker \
  --trace data/example_unweighted.json \
  --capacity 3 \
  --derive-predicted-caches
```

### Local `evict_value_v1` first check (small; not the Wulver `heavy_r1` line)

```bash
python scripts/build_evict_value_dataset_v1.py --max-rows 200000
python scripts/train_evict_value_v1.py --horizon 8
python scripts/run_evict_value_v1_first_check.py
```

---

## Policy families (`run_policy` registry)

### Literature baselines and robust references

`lru`, `weighted_lru`, `advice_trusting`, `la_det`, `la_det_approx`, `la_det_faithful`, `marker`, `blind_oracle`, `predictive_marker`, `adaptive_query` (`parsimonious_caching`), `trust_and_doubt`, `robust_ftp_d_marker` (`robust_ftp`), `blind_oracle_lru_combiner`, `offline_belady`

### Experimental policies

`atlas_v1`, `atlas_v2`, `atlas_v3`, `atlas_cga_v1` (`atlas_cga`), `atlas_cga_v2`, `rest_v1`, `ml_gate_v1`, `ml_gate_v2`, `evict_value_v1`, `evict_value_v1_guarded`, `sentinel_robust_tripwire_v1`, `sentinel_budgeted_guard_v2`

**Pairwise learned line** (separate scripts, not in `POLICY_REGISTRY`): `scripts/build_evict_value_pairwise_dataset.py`, `scripts/train_evict_value_pairwise_v1.py`, `scripts/run_evict_value_pairwise_first_check.py`.

Details: `docs/baselines.md`, `docs/framework.md`.

---

## Datasets

```bash
python scripts/datasets/prepare_all.py --dataset <brightkite|citibike|spec_cpu2006|wiki2018|twemcache|metakv|metacdn|cloudphysics|all>
```

- Raw: `data/raw/<dataset>/` — Processed: `data/processed/<dataset>/` — Notes: `docs/datasets.md`

---

## Other experiment families (not the canonical KBS `heavy_r1` path unless labeled)

> Sections A–D are useful entry points; they are **not** interchangeable with the Wulver `heavy_r1` manuscript pipeline.

### A) Offline-teacher vs heuristic

```bash
python scripts/run_offline_teacher_vs_heuristic_experiment.py \
  --trace-glob "data/example_*.json,data/example_general_caching.json" \
  --capacities 2,3 --horizon 12 \
  --output-dir analysis/offline_teacher_vs_heuristic
```

See `docs/offline_teacher_vs_heuristic_mediumscale.md`.

### B) Pairwise vs pointwise

```bash
python scripts/run_pairwise_vs_pointwise_experiment.py --output-dir analysis/pairwise_vs_pointwise
```

Interpret conservatively: `docs/pairwise_vs_pointwise_experiment.md`, `docs/manuscript_evidence_map.md`.

### C) Sentinel / guard refinement

```bash
python scripts/run_sentinel_budgeted_guard_v2_eval.py
python scripts/run_sentinel_budgeted_guard_v2_ablation.py
```

---

## Output conventions

Default roots: **`analysis/`** (experiments and manuscript-support), **`output/`** (ad hoc). New work should use `analysis/<experiment_name>/` with `summary.json` + `report.md` when possible.

---

## Testing

```bash
pytest tests/ -v
```

---

## Navigation and hygiene

| Topic | Document |
|-------|----------|
| Cleanup / navigation audit (this release) | [`docs/repository_cleanup_report.md`](docs/repository_cleanup_report.md) |
| KBS hygiene notes | [`docs/kbs_repository_hygiene_report.md`](docs/kbs_repository_hygiene_report.md) |
| Exploratory lightweight ablations | [`docs/lightweight_exploratory_ablations.md`](docs/lightweight_exploratory_ablations.md) |
