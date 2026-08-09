# Local to Wulver Continuation Sync Manifest

Status: `LOCAL_TO_WULVER_SYNC_REQUIRED = YES`

Do not sync generated smoke outputs or local analysis directories as part of
this package. The full experiment should be launched on Wulver only after these
source/protocol files are present there and the frozen pi1 registry/model paths
are verified.

## Files to Sync

Source:

- `src/lafc/continuation_policy_ablation.py`
- `scripts/experiments/run_continuation_policy_causal_ablation_smoke.py`

Tests:

- `tests/test_continuation_policy_ablation.py`

Config:

- `configs/continuation_policy_causal_ablation_v1.json`

Documentation:

- `docs/reviewer/kbs_negative_results_interpretation.md`
- `docs/reviewer_revision_roadmap.md`
- `docs/reviewer/kbs_second_revision_artifact_map.md`
- `docs/reviewer/local_to_wulver_continuation_sync_manifest.md`

## Existing Required Wulver Inputs

These are not new files from this task, but the Wulver run must verify them
before launch:

- `analysis/supervision_objective_ablation_v1/model_registry.json`
- `models/supervision_objective_ablation_v1/objective_eviction_loss/<held_out_family>.pkl`
- `configs/fair_cross_family_v1/folds/*.json`
- `configs/fair_cross_family_v1/folds/*_family_split_map.json`
- each fold's train manifest and held-out trace paths referenced from the fold
  configs

## Expected Local Hashes

Record these after the final local validation pass and before sync:

```bash
sha256sum \
  src/lafc/continuation_policy_ablation.py \
  scripts/experiments/run_continuation_policy_causal_ablation_smoke.py \
  tests/test_continuation_policy_ablation.py \
  configs/continuation_policy_causal_ablation_v1.json \
  docs/reviewer/kbs_negative_results_interpretation.md \
  docs/reviewer_revision_roadmap.md \
  docs/reviewer/kbs_second_revision_artifact_map.md \
  docs/reviewer/local_to_wulver_continuation_sync_manifest.md
```

## Launch Preconditions

- Wulver branch matches or intentionally incorporates the local commits.
- Frozen pi1 registry has `MODEL_SELECTION_FROZEN=true`.
- Each pi1 artifact hash matches the registry.
- Held-out family is absent from pi1 training and validation for every fold.
- C1 and C2 dataset construction emits identical `(decision_id, candidate_id)`
  pairs.
- No fallback pi1 model is allowed.
- Full campaign uses `configs/continuation_policy_causal_ablation_v1.json`.

## Out of Scope for Sync

- local smoke stdout
- `/tmp` smoke artifacts
- `analysis/continuation_policy_causal_ablation*`
- unrelated oracle, fairness, or target-degeneracy generated outputs
- active tmux session logs
