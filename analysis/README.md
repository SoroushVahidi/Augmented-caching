# Analysis artifacts guide

`analysis/` stores experiment outputs and manuscript-support artifacts.

## What is canonical vs exploratory?

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

### Stable helper artifacts

These are often consumed by scripts or runbooks:
- `analysis/wulver_trace_manifest*.csv`
- `analysis/evict_value_failure_slice_audit.csv`
- `analysis/evict_value_failure_slice_summary.md`

### Legacy root-level single-file outputs

Files such as `*_first_check.csv`, `*_first_check.md`, and older model-comparison tables remain for traceability.
New experiments should prefer dedicated subdirectories.

## Naming conventions (for new runs)

- Use `analysis/<experiment_name>/` as default `--output-dir`.
- Use explicit names: `results.csv`, `summary.json`, `report.md`.
- Keep one `summary.json` per output directory for scripting.
- If a markdown summary is separate from `report.md`, use `<experiment_name>.md` sparingly and only when backward compatibility requires it.

## Why this repository keeps older files

Some manuscript/proof/audit docs and scripts directly reference historical artifacts.
To preserve reproducibility and citation continuity, this repository retains older files instead of aggressively deleting or renaming them.
