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

Current local SHA256 hashes for the continuation sync package:

| File | SHA256 |
|---|---|
| `src/lafc/continuation_policy_ablation.py` | `a5bdc14ceb9269969f570860ac3a7e34ef75c9c3b4efd4dc1d1d0bd27b0b4701` |
| `scripts/experiments/run_continuation_policy_causal_ablation_smoke.py` | `f44ff9974784debb836fbac0ad7e601c691b0f60ec416e47ace643fbcd00d87a` |
| `tests/test_continuation_policy_ablation.py` | `c429a8fd957da005d0d0abdbe09ebaa980b4baab8a8407174b83653e805bca60` |
| `configs/continuation_policy_causal_ablation_v1.json` | `881210c980b3d5615914823a5aa08922f02dd8e660f0dc9c7d7b180ef9ee2400` |
| `docs/reviewer/kbs_negative_results_interpretation.md` | `c5e3420e5223546dc8e5bd65480d7c8d9ea506250195505b5eb6f1f11a1cce1b` |
| `docs/reviewer_revision_roadmap.md` | `e6dbeb143dec1a1eb498912974fa2cceacfd3be4790be61b4ae101f448b86cb1` |
| `docs/reviewer/kbs_second_revision_artifact_map.md` | `0ca4c4ca91205e2973bd0fa4eea306f7cbd8f4f31939189ad4164db7faa28b5a` |
| `docs/reviewer/local_to_wulver_continuation_sync_manifest.md` | Self-hash is not stable when embedded; compute at transfer time with `sha256sum`. |

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
