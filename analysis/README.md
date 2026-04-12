# Analysis artifacts guide

`analysis/` stores experiment outputs and manuscript-support artifacts.

**Navigation:** For the finalized **KBS** manuscript (`heavy_r1` Wulver line, builder command, `tables/manuscript/` / `figures/manuscript/`), start at **[`../CANONICAL_KBS_SUBMISSION.md`](../CANONICAL_KBS_SUBMISSION.md)** and **`docs/kbs_manuscript_workflow.md`**.

---

## How this directory is organized (taxonomy)

| Kind | Typical location | Citable as KBS main Wulver evidence? |
|------|------------------|--------------------------------------|
| **Canonical `heavy_r1`** | Root-level `*_heavy_r1.{csv,json,md}` per `docs/evict_value_v1_kbs_canonical_artifacts.md` | **Yes** — when using those exact filenames |
| **Wiring / smoke** | `*_heavy_smoke.*` (e.g. policy comparison smoke CSV) | **No** — Slurm smoke only; not the full manifest eval |
| **Legacy / alternate drivers** | Unsuffixed `evict_value_wulver_v1_*` at repo root of `analysis/` | **No** for KBS main claims (extra policies / wrong driver) |
| **Stable experiment dirs** | `analysis/<name>/` with `report.md`, `summary.json`, CSVs | **Only** if that experiment is explicitly claimed in the paper |
| **Exploratory campaigns** | `analysis/pairwise_*_campaign/` (large `jobs/` trees) | **No** unless cross-walked from manuscript text |
| **Lightweight ablations** | `analysis/*_light/` | **No** — see `docs/lightweight_exploratory_ablations.md` |
| **Manifests / audits** | `analysis/wulver_trace_manifest*.csv`, `analysis/evict_value_failure_slice_*` | Helper / audit — cite role explicitly |

**We do not delete** legacy or exploratory files; they remain for provenance and older references.

---

## KBS / Wulver `evict_value_v1`: canonical vs non-canonical root files

For the **Knowledge-Based Systems** manuscript, **only** the **`heavy_r1`-suffixed** analysis files below are inputs to `scripts/paper/build_kbs_main_manuscript_artifacts.py` (see `docs/evict_value_v1_kbs_canonical_artifacts.md`). Other similarly named files may remain in this directory **for history**; do not treat them as interchangeable without checking provenance.

| Canonical (KBS main Wulver line) | Non-canonical (historical / alternate drivers — do not cite as main KBS comparison) |
|-----------------------------------|-------------------------------------------------------------------------------------|
| `evict_value_wulver_v1_policy_comparison_heavy_r1.csv` | `evict_value_wulver_v1_policy_comparison.csv` |
| `evict_value_wulver_v1_policy_comparison_heavy_r1.md` | `evict_value_wulver_v1_policy_comparison.md` |
| `evict_value_wulver_v1_train_metrics_heavy_r1.json` | `evict_value_wulver_v1_train_metrics.json` |
| `evict_value_wulver_v1_model_comparison_heavy_r1.csv` | `evict_value_wulver_v1_model_comparison.csv` |
| `evict_value_wulver_v1_best_config_heavy_r1.json` | `evict_value_wulver_v1_best_config.json` |
| `evict_value_v1_wulver_dataset_summary_heavy_r1.md` | `evict_value_v1_wulver_dataset_summary.md` |

**We do not delete** non-canonical files here; they support older references and alternate experiment drivers. Treat unsuffixed names as **stale or alternate-driver** for KBS main claims unless proven identical to the `*_heavy_r1` file.

## What is canonical vs exploratory?

### Canonical KBS Wulver manuscript-safe artifacts (strict)

Treat **only** these as canonical for the main Wulver `evict_value_v1` paper line:
- `analysis/*_heavy_r1.*` files listed in `docs/evict_value_v1_kbs_canonical_artifacts.md`
- manuscript bundle outputs generated from those files:
  - `tables/manuscript/`
  - `figures/manuscript/`
  - `reports/manuscript_artifacts/`

If `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` is missing, the canonical line is incomplete.

### Canonical experiment-style outputs (preferred)

These live in dedicated subdirectories and usually include:
- `results.csv` or `policy_comparison.csv`
- `report.md`
- `summary.json`

Examples:
- `analysis/pairwise_vs_pointwise/`
- `analysis/offline_teacher_vs_heuristic_mediumscale/`
- `analysis/sentinel_budgeted_guard_v2/`
- `analysis/hybrid_fallback_experiment/`

### Manuscript bundle outputs (generated for LaTeX)

When present, these are the curated submission-oriented copies (sources for tables/figures may still live under `analysis/`):

- `tables/manuscript/` — CSV + `.tex` fragments
- `figures/manuscript/` — PDF + PNG
- `reports/manuscript_artifacts/` — manifest, narrative report, optional `latex_snippets/`

**evict_value_v1 / KBS:** The generator `scripts/paper/build_kbs_main_manuscript_artifacts.py` reads **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`** (and other `*_heavy_r1` training artifacts), not the unsuffixed `evict_value_wulver_v1_policy_comparison.csv`. See `docs/evict_value_v1_kbs_canonical_artifacts.md`.

See `docs/reproducibility_and_artifacts.md` for how these relate to exploratory campaign trees (for example large `analysis/pairwise_*_campaign/jobs/` directories).

### Stable helper artifacts

These are often consumed by scripts or runbooks:
- `analysis/wulver_trace_manifest*.csv`
- `analysis/evict_value_failure_slice_audit.csv`
- `analysis/evict_value_failure_slice_summary.md`

### Legacy root-level single-file outputs

Files such as `*_first_check.csv`, `*_first_check.md`, and older model-comparison tables remain for traceability.
New experiments should prefer dedicated subdirectories.

### Exploratory lightweight ablations (grouped, non-canonical)

These are useful and intentionally retained, but they are **not** the main KBS Wulver path:
- `analysis/incoming_file_aware_ablation_light/` (incoming-aware)
- `analysis/history_context_ablation_light/` (history-aware)
- `analysis/history_objective_ablation_light/` (history-objective / history-pairwise-style objective)
- `analysis/joint_cache_state_model_light/` (joint cache-state modeling)
- `analysis/joint_state_reasoning_light/` (joint-state reasoning)

Script index for these families: `docs/lightweight_exploratory_ablations.md`.

### Pairwise / ranking campaigns and large `jobs/` trees (exploratory)

Outputs under `analysis/pairwise_*_campaign/` (including per-task `jobs/<label>/`) are **campaign artifacts** for publishability, chain-witness, or related studies. They are **not** the canonical `heavy_r1` Wulver `evict_value_v1` comparison unless explicitly cross-walked in manuscript docs. Slurm: `slurm/pairwise_*_campaign_*.sbatch`. See `docs/manuscript_evidence_map.md`, `docs/wulver_pairwise_publishability_campaign.md`, `docs/wulver_pairwise_chain_witness_campaign.md`.

### Theorem / proof / audit support (exploratory)

Scripts such as `scripts/search_*`, `scripts/analyze_*`, and docs such as `docs/pairwise_theory_roadmap.md` / `analysis/pairwise_inversion_examples.md` support **development and audits**, not the canonical KBS Wulver table inputs. Cite separately from `*_heavy_r1` analysis files.

## Naming conventions (for new runs)

- Use `analysis/<experiment_name>/` as default `--output-dir`.
- Use explicit names: `results.csv`, `summary.json`, `report.md`.
- Keep one `summary.json` per output directory for scripting.
- If a markdown summary is separate from `report.md`, use `<experiment_name>.md` sparingly and only when backward compatibility requires it.

## Why this repository keeps older files

Some manuscript/proof/audit docs and scripts directly reference historical artifacts.
To preserve reproducibility and citation continuity, this repository retains older files instead of aggressively deleting or renaming them.
