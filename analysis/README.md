# Analysis guide

This directory contains experimental results, evaluation metrics, and manuscript-support artifacts.

## Directory structure

- `analysis/manuscript_canonical/` — **Tracked**: Frozen results for the final KBS manuscript (e.g., `heavy_r1` evaluation).
- `analysis/reviewer_revision/` — **Tracked**: Results from reviewer-revision experiments (e.g., fairness, objective-ablation).
- `analysis/exploratory/` — **Not Tracked**: First-check runners, ad-hoc ablations, and exploratory diagnostics.
- `analysis/diagnostics/` — **Not Tracked**: Failure-slice audits and overhead benchmarks.
- `analysis/manifests/` — **Tracked**: Trace manifests used to define experiment splits.
- `analysis/external_learned_baselines/` — **Not Tracked**: Results from large-scale external baseline evaluations (e.g., LRB).

## Version Control Policy

- **Tracked**: Frozen canonical results, small summary tables, and authoritative provenance JSONs.
- **Not Tracked**: Large per-step decision logs, temporary sweep results, and redundant CSVs.

## KBS manuscript workflow

The canonical **`heavy_r1`** submission path uses artifacts in `analysis/manuscript_canonical/`. Refer to `docs/kbs_manuscript_workflow.md` for details.
