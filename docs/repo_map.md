# Repository map

This document is a concise orientation guide for external readers.

## KBS second-revision workflow

For the current reviewer-science branch and its four reviewer concerns:

- `docs/kbs_manuscript_workflow.md`
- `docs/kbs_second_revision_repository_state.md`
- `docs/reviewer/kbs_second_revision_artifact_map.md`
- `docs/reviewer/kbs_evidence_eligibility.md`
- `docs/reviewer/kbs_negative_results_interpretation.md`

Historical Wulver `heavy_r1` material is still preserved, but it is no longer
the best top-level orientation for the current cleanup branch. When needed:

- historical checklist hub: `CANONICAL_KBS_SUBMISSION.md`
- historical runbook: `docs/wulver_heavy_evict_value_experiment.md`
- historical artifact set: `docs/evict_value_v1_kbs_canonical_artifacts.md`

## Top-level layout

- `src/lafc/` — core library implementation (policies, simulator, runners, datasets, offline solvers).
- `scripts/` — reproducible experiment and dataset-prep entry points.
- `tests/` — unit and integration tests.
- `docs/` — protocols, workflow notes, artifact maps, and scientific notes.
- `analysis/` — generated text artifacts from experiments plus small tracked audits.
- `data/` — small examples in git plus raw and derived data roots.
- `slurm/` — cluster batch templates for heavier runs, including historical
  `heavy_r1` jobs and newer reviewer-science orchestration templates.

## `src/lafc/` subpackages

- `policies/` — policy implementations.
- `simulator/` — cache state and trace execution logic.
- `runner/` — CLI entrypoint (`python3 -m lafc.runner.run_policy`).
- `datasets/` — dataset ingestion and preprocessing helpers.
- `offline/` — offline reference solvers and helpers.
- `learned_gate/` — gate datasets, features, and models.
- top-level `evict_*` modules — learned eviction datasets and models.
- `metrics/` — common cost and prediction metrics.

## `scripts/` families

- `scripts/experiments/` — reviewer-science runners and audits plus focused ablations.
- `scripts/validation/` — read-only repository and campaign status helpers.
- `scripts/paper/` — historical manuscript bundle builders, including the
  older `heavy_r1` KBS path.
- `scripts/datasets/` — dataset download and prepare wrappers.
- root-level `build_*`, `train_*`, `run_*`, `search_*`, `analyze_*` —
  experiment-specific entry points kept for backward compatibility.

## `analysis/` conventions

- experiment directories are preferred,
- historical root-level artifacts are retained for provenance,
- many reviewer outputs are intentionally preserved locally but not tracked,
- the eligibility note determines which outputs are usable for which claims.

See `analysis/README.md` and
`docs/reviewer/kbs_evidence_eligibility.md`.

## Read first

1. `docs/kbs_manuscript_workflow.md`
2. `docs/reviewer/kbs_second_revision_artifact_map.md`
3. `docs/reviewer/kbs_evidence_eligibility.md`
4. `docs/kbs_second_revision_repository_state.md`
5. `docs/reproducibility_and_artifacts.md`
6. `docs/baselines.md`
7. `docs/framework.md`
