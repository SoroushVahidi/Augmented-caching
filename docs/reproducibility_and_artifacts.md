# Reproducibility: entry points, artifacts, and manuscript vs exploratory outputs

This document orients readers who need to **rerun** experiments or **cite** outputs without overstating what the repository proves.

## Primary simulator CLI

The main online policy simulator is:

```bash
python -m lafc.runner.run_policy --policy <name> --trace <path> --capacity <k>
```

- **Trace formats:** JSON or CSV via `load_trace` (see `lafc.simulator.request_trace`). Some dataset pipelines emit `.jsonl` under `data/processed/`; heavier experiment code paths may load them via helpers such as `load_trace_from_any` in `lafc.evict_value_wulver_v1`. For ad hoc `run_policy` runs, use JSON/CSV unless you add a compatible loader.
- **Policy names:** exactly the keys of `POLICY_REGISTRY` in `src/lafc/runner/run_policy.py` (also listed in the module docstring and in `README.md`).
- **Not in this CLI:** `evict_value_pairwise_v1` (requires a trained artifact; use `scripts/train_evict_value_pairwise_v1.py` and `scripts/run_evict_value_pairwise_first_check.py` or related `run_*pairwise*` scripts). Offline **general caching** (variable sizes/costs) uses `scripts/run_offline_general_caching_approx.py` (solver label `offline_general_caching_lp_round`).

## Canonical experiment paths (high level)

| Goal | Typical commands | Default / typical output roots |
|------|------------------|--------------------------------|
| **KBS main Wulver `evict_value_v1` (`heavy_r1` only)** | `slurm/evict_value_v1_wulver_heavy_train.sbatch` then `slurm/evict_value_v1_wulver_heavy_eval.sbatch`; see `docs/wulver_heavy_evict_value_experiment.md` | `analysis/*_heavy_r1.*`, `models/evict_value_wulver_v1_best_heavy_r1.pkl`; then `scripts/paper/build_kbs_main_manuscript_artifacts.py` → `tables/manuscript/`, etc. |
| Pointwise learned eviction (local / first-check) | `scripts/build_evict_value_dataset_v1.py`, `scripts/train_evict_value_v1.py`, `scripts/run_evict_value_v1_first_check.py` | `analysis/evict_value_v1_*`, `models/` |
| Pairwise learned eviction (`evict_value_pairwise_v1`) | `scripts/build_evict_value_pairwise_dataset.py`, `scripts/train_evict_value_pairwise_v1.py`, `scripts/run_evict_value_pairwise_first_check.py` | `analysis/evict_value_pairwise_*`, `models/` |
| Offline teacher vs heuristic | `scripts/run_offline_teacher_vs_heuristic_experiment.py` | `analysis/offline_teacher_vs_heuristic/` or `--output-dir` |
| Pairwise vs pointwise | `scripts/run_pairwise_vs_pointwise_experiment.py` | `analysis/pairwise_vs_pointwise/` |
| Datasets | `scripts/datasets/prepare_all.py` | `data/raw/<dataset>/`, `data/processed/<dataset>/` |
| Manuscript table/figure bundle | `scripts/paper/build_kbs_main_manuscript_artifacts.py` | `tables/manuscript/`, `figures/manuscript/`, `reports/manuscript_artifacts/` |

**Index:** `docs/kbs_manuscript_submission_index.md`.

For **evict_value_v1** KBS evidence, inputs are **`heavy_r1` files only** (see `docs/evict_value_v1_kbs_canonical_artifacts.md` and `docs/wulver_heavy_evict_value_experiment.md`). The builder does **not** use the unsuffixed `analysis/evict_value_wulver_v1_policy_comparison.csv`.

For heavier cluster runs, see `slurm/*.sbatch` and the Wulver runbooks under `docs/wulver_*.md` and `docs/evict_value_v1_wulver_runbook.md`.

## Manuscript-safe vs exploratory

**Manuscript-oriented bundles (curated for LaTeX inclusion):**

- `tables/manuscript/` — CSV + `.tex` snippets generated for the paper package.
- `figures/manuscript/` — PDF + PNG figures intended for submission.
- `reports/manuscript_artifacts/` — manifest, report, and `latex_snippets/` helpers.

These are **not** by themselves scientific claims; they summarize repository-generated numbers. Interpretation caveats live in `docs/manuscript_evidence_map.md` and `docs/manuscript_open_questions.md`.

**Exploratory / research-active (use with explicit caveats):**

- `docs/pairwise_*` (theorem sketches, audits, attacks) — development artifacts; not finished proofs.
- Large per-job trees under `analysis/pairwise_*_campaign/jobs/` — empirical campaign outputs; may include large binary artifacts (for example trained checkpoints). Prefer citing aggregate CSV/JSON summaries when possible.

**Stable helper files** (referenced by scripts and audits; keep under version control when small):

- Wulver trace manifests under `analysis/wulver_trace_manifest*.csv`
- Failure-slice audits such as `analysis/evict_value_failure_slice_*.csv` / `.md`

**Legacy root-level `analysis/*.csv` / `*.md`:** retained for history and backward compatibility; new work should prefer `analysis/<experiment_name>/` with `summary.json` + `report.md`.

## Conservative framing

When describing results, distinguish: (1) **implemented baselines** with literature pointers in `docs/baselines.md`, (2) **experimental** policies in `docs/framework.md`, and (3) **learned** policies that depend on training data and model seeds. Avoid implying proved competitive guarantees unless a doc explicitly states a checked theorem with assumptions.
