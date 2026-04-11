# Learning-Augmented Caching (`lafc`)

A research codebase for learning-augmented caching algorithms, robust baselines, and manuscript-support experiments.

This repository includes:
- faithful and robust policy baselines from the LA-caching literature,
- experimental learned eviction policies and guard wrappers,
- dataset-preparation pipelines for public and manifest-based traces,
- reproducible analysis scripts that emit text artifacts (`.csv`, `.json`, `.md`).

## Project status (conservative snapshot)

- **Most stable, manuscript-safe references:** classical baselines (`lru`, `marker`, `predictive_marker`), robust combiners (`robust_ftp_d_marker`, `blind_oracle_lru_combiner`), and the unweighted offline reference `offline_belady` (via `python -m lafc.runner.run_policy`). For **general caching** (variable sizes/costs), the LP+rounding offline baseline is run separately; see `scripts/run_offline_general_caching_approx.py` and `docs/offline_general_caching_approx.md` (not a `--policy` value on the main simulator CLI).
- **Main learned line in this repo:** `evict_value_v1` (candidate-level scoring) plus guarded/fallback variants.
- **Research-active exploratory lines:** pairwise/ranking supervision, sentinel mechanisms, and theorem-development notes under `docs/pairwise_*`.

For evidence-strength caveats and what is still open, see `docs/manuscript_evidence_map.md` and `docs/manuscript_open_questions.md`.

### KBS manuscript: canonical Wulver `heavy_r1` path (**main paper path**)

For the **Knowledge-Based Systems** submission package, the repository treats **one** end-to-end line as authoritative for main **multi-trace Wulver** `evict_value_v1` numbers:

1. **Runbook:** `docs/wulver_heavy_evict_value_experiment.md`
2. **Cluster drivers:** `slurm/evict_value_v1_wulver_heavy_train.sbatch`, `slurm/evict_value_v1_wulver_heavy_eval.sbatch` (`EXP_TAG=heavy_r1`)
3. **Exact input/output filenames:** `docs/evict_value_v1_kbs_canonical_artifacts.md`
4. **LaTeX-oriented bundle:** `scripts/paper/build_kbs_main_manuscript_artifacts.py` → `tables/manuscript/`, `figures/manuscript/`, `reports/manuscript_artifacts/`

**Do not** treat the unsuffixed `analysis/evict_value_wulver_v1_policy_comparison.csv` as the main KBS comparison (it may include extra policies from non-heavy drivers). See `analysis/README.md` and `docs/kbs_manuscript_submission_index.md`.

### One-screen canonical checklist (after heavy eval completes)

Use this exact sequence to avoid mixing canonical and exploratory artifacts:

```bash
# 1) verify canonical heavy_r1 eval output exists
test -f analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv

# 2) build manuscript bundle from heavy_r1-only evidence
python scripts/paper/build_kbs_main_manuscript_artifacts.py

# 3) inspect generated submission artifacts
ls tables/manuscript figures/manuscript reports/manuscript_artifacts
```

If step (1) fails, run/finish the heavy eval job from `docs/wulver_heavy_evict_value_experiment.md` first.

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Quick start

### Run a baseline policy

```bash
python -m lafc.runner.run_policy \
  --policy predictive_marker \
  --trace data/example_unweighted.json \
  --capacity 3
```

### Run a robust combiner baseline

```bash
python -m lafc.runner.run_policy \
  --policy robust_ftp_d_marker \
  --trace data/example_unweighted.json \
  --capacity 3 \
  --derive-predicted-caches
```

### Run the current main learned policy (`evict_value_v1`)

```bash
python scripts/build_evict_value_dataset_v1.py --max-rows 200000
python scripts/train_evict_value_v1.py --horizon 8
python scripts/run_evict_value_v1_first_check.py
```

---

## Policy families

### Literature baselines and robust references (`python -m lafc.runner.run_policy`)

- `lru`, `weighted_lru`, `advice_trusting`, `la_det`, `la_det_approx`, `la_det_faithful`
- `marker`, `blind_oracle`, `predictive_marker`
- `adaptive_query` (`parsimonious_caching` alias)
- `trust_and_doubt`
- `robust_ftp_d_marker` (`robust_ftp` alias)
- `blind_oracle_lru_combiner`
- `offline_belady` (unweighted offline optimum on the trace; requires full lookahead via trace construction)

**General-caching offline baseline (separate entry point):** LP relaxation + deterministic rounding via `scripts/run_offline_general_caching_approx.py` (documented as the “offline general caching approx” family in `docs/offline_general_caching_approx.md`). The internal solver name is `offline_general_caching_lp_round`; it is **not** selected with `--policy` on `run_policy`.

### Experimental policies (`run_policy` unless noted)

- `atlas_v1`, `atlas_v2`, `atlas_v3`
- `atlas_cga_v1` (`atlas_cga` alias), `atlas_cga_v2`
- `rest_v1`
- `ml_gate_v1`, `ml_gate_v2`
- `evict_value_v1`, `evict_value_v1_guarded` (artifact path via `--evict-value-model-path`)
- `sentinel_robust_tripwire_v1`, `sentinel_budgeted_guard_v2`

**Learned pairwise line (scripts, not `run_policy`):** `evict_value_pairwise_v1` is trained/evaluated through `scripts/build_evict_value_pairwise_dataset.py`, `scripts/train_evict_value_pairwise_v1.py`, and `scripts/run_evict_value_pairwise_first_check.py` (and related `run_*pairwise*` tools). It does not appear in `POLICY_REGISTRY` because it requires a trained artifact.

See `docs/baselines.md`, `docs/framework.md`, and `docs/reproducibility_and_artifacts.md` for algorithm notes, runnable entry points, and manuscript vs exploratory outputs.

---

## Datasets

Supported dataset families include BrightKite, CitiBike, SPEC CPU2006, wiki2018, and manifest-based production-style traces (`twemcache`, `metakv`, `metacdn`, `cloudphysics`).

Prepare all or one dataset:

```bash
python scripts/datasets/prepare_all.py --dataset <brightkite|citibike|spec_cpu2006|wiki2018|twemcache|metakv|metacdn|cloudphysics|all>
```

- Raw inputs: `data/raw/<dataset>/`
- Processed outputs: `data/processed/<dataset>/`
- Format and legal/source notes: `docs/datasets.md`

---

## Reproducing main experiment families

> **Scope note:** sections A–D below are useful experiment entry points, but they are **not** the canonical KBS Wulver `heavy_r1` manuscript pipeline unless explicitly labeled as such.

### A) `evict_value_v1` first check

```bash
python scripts/build_evict_value_dataset_v1.py --max-rows 200000
python scripts/train_evict_value_v1.py --horizon 8
python scripts/run_evict_value_v1_first_check.py
```

Outputs (default):
- `analysis/evict_value_v1_first_check.csv`
- `analysis/evict_value_v1_first_check.md`
- `analysis/evict_value_v1_metrics.json`
- `analysis/evict_value_v1_model_comparison.csv`

### B) Offline-teacher vs heuristic supervision

```bash
python scripts/run_offline_teacher_vs_heuristic_experiment.py \
  --trace-glob "data/example_*.json,data/example_general_caching.json" \
  --capacities 2,3 \
  --horizon 12 \
  --output-dir analysis/offline_teacher_vs_heuristic
```

For the stronger controlled run, see `docs/offline_teacher_vs_heuristic_mediumscale.md`.

### C) Pairwise vs pointwise supervision

```bash
python scripts/run_pairwise_vs_pointwise_experiment.py \
  --output-dir analysis/pairwise_vs_pointwise
```

Interpret conservatively; see `docs/pairwise_vs_pointwise_experiment.md` and `docs/manuscript_evidence_map.md`.

### D) Sentinel/guard refinement

```bash
python scripts/run_sentinel_budgeted_guard_v2_eval.py
python scripts/run_sentinel_budgeted_guard_v2_ablation.py
```

---

## Output conventions

Default output root is `analysis/` for manuscript-support artifacts and `output/` for ad hoc runs.

Typical files:
- `summary.json` (aggregate metrics)
- `report.md` (human-readable interpretation)
- `*.csv` (slice-level tables)
- `*_summary.json` (compact machine-readable summary)

See `analysis/README.md` for canonical vs legacy/exploratory organization.

---

## Repository guide

- KBS submission index (canonical `heavy_r1`, tables/figures, non-canonical notes): `docs/kbs_manuscript_submission_index.md`
- High-level map: `docs/repo_map.md`
- Reproducibility, CLI entry points, manuscript vs exploratory artifacts: `docs/reproducibility_and_artifacts.md`
- Analysis artifact organization: `analysis/README.md`
- Script organization and naming conventions: `scripts/README.md`
- Lightweight exploratory ablations index (incoming-aware/history-aware/history-pairwise/joint-state): `docs/lightweight_exploratory_ablations.md`

---

## Testing

```bash
pytest tests/ -v
```
