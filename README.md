# Learning-Augmented Caching (`lafc`)

`lafc` is a research codebase for learning-augmented caching: cache
simulation, literature-faithful baselines, learned eviction policies, dataset
pipelines, and reproducible experiment drivers.

## Current repository orientation

As of August 9, 2026, the main local reviewer-science branch is
`kbs/second-revision-science`. It preserves:

- the current KBS second-revision code and audit tooling,
- earlier historical `heavy_r1` Wulver manuscript material,
- multiple generations of exploratory learned-caching experiments.

The current branch documents a mostly negative or methodological result:
`evict_value_v1` is scientifically interesting, but the second-revision work is
about understanding why a seemingly reasonable offline target performs poorly
in closed-loop deployment, not about claiming a new performance SOTA.

## What is here

- `src/lafc/`: simulator, policies, runners, dataset helpers, offline solvers.
- `scripts/`: reproducible entry points for experiments and validation.
- `tests/`: unit and integration tests.
- `docs/`: protocols, artifact maps, workflow notes, and manuscript-support docs.
- `analysis/`: generated outputs and small tracked audits.
- `slurm/`: batch templates for heavier runs.

More detail: [docs/repo_map.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/repo_map.md).

## Current reviewer-science entry points

- Repository state and tracked-vs-generated boundaries:
  [docs/kbs_second_revision_repository_state.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/kbs_second_revision_repository_state.md)
- Current KBS second-revision workflow:
  [docs/kbs_manuscript_workflow.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/kbs_manuscript_workflow.md)
- Reviewer artifact map:
  [docs/reviewer/kbs_second_revision_artifact_map.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/reviewer/kbs_second_revision_artifact_map.md)
- Evidence eligibility rules:
  [docs/reviewer/kbs_evidence_eligibility.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/reviewer/kbs_evidence_eligibility.md)
- Internal note on the negative results:
  [docs/reviewer/kbs_negative_results_interpretation.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/reviewer/kbs_negative_results_interpretation.md)
- Read-only campaign status tools:
  `python3 scripts/validation/revision_status.py`
  and
  `python3 scripts/validation/revision_readiness.py`

Reviewer experiment drivers and audits live mainly under `scripts/experiments/`.

## Generated evidence vs tracked source

Tracked in Git:

- code under `src/lafc/`,
- experiment and validation entry points under `scripts/`,
- tests under `tests/`,
- protocols, artifact maps, and notes under `docs/`,
- small canonical audits and fixtures.

Preserved locally but intentionally not curated as tracked release material:

- most reviewer result CSV/JSON/MD outputs under `analysis/`,
- model artifacts under `models/`,
- large derived datasets under `data/derived/`.

The artifact map and eligibility note above are the source of truth for which
generated outputs are complete, partial, smoke-only, contaminated, historical,
or currently usable for reviewer-facing tables.

## Basic setup

```bash
pip install -e ".[dev]"
```

## Basic tests

```bash
PYTHONPATH=src pytest tests/ -v
```

Optional dependencies:

- `lightgbm` is required for some learned-baseline tests and policies such as
  `lrb` and `three_l_cache`.

## Basic simulator usage

```bash
python3 -m lafc.runner.run_policy \
  --policy lru \
  --trace data/example_unweighted.json \
  --capacity 3
```

## Reviewer experiment families

- Learned-baseline fairness comparisons and held-out retraining:
  `scripts/experiments/run_reviewer_fairness.py`,
  `scripts/experiments/run_evict_cross_family_heldout_eval.py`
- Supervision-objective ablation:
  `scripts/experiments/run_supervision_objective_ablation_eval.py`
  plus the audit and gate scripts in the same directory
- Distribution-shift diagnosis:
  `scripts/experiments/run_distribution_shift_ablation.py`
- Practical-significance analysis:
  `scripts/experiments/run_practical_significance_ablation.py`

These write generated outputs under `analysis/<experiment>/` or related
reviewer-specific paths documented in
[docs/reviewer/kbs_second_revision_artifact_map.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/reviewer/kbs_second_revision_artifact_map.md).

## Historical heavy-run material

The earlier Wulver `heavy_r1` manuscript path is still preserved, but it is no
longer the best top-level orientation for this cleanup pass. It remains useful
as historical provenance and for older manuscript-support builders.

- Historical heavy-run workflow:
  [docs/wulver_heavy_evict_value_experiment.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/wulver_heavy_evict_value_experiment.md)
- Historical heavy-run artifact set:
  [docs/evict_value_v1_kbs_canonical_artifacts.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/evict_value_v1_kbs_canonical_artifacts.md)

## Documentation index

[docs/README.md](/home/soroush/Augmented-caching-kbs-second-revision/docs/README.md)
