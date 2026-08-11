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

## Expected Local Hashes -- pinned payload (source / tests / config)

These files are the actual sync payload whose content must transfer
byte-for-byte. Their hashes are pinned and re-verified stable as of
2026-08-10 (no drift since first recorded):

| File | SHA256 |
|---|---|
| `src/lafc/continuation_policy_ablation.py` | `a5bdc14ceb9269969f570860ac3a7e34ef75c9c3b4efd4dc1d1d0bd27b0b4701` |
| `scripts/experiments/run_continuation_policy_causal_ablation_smoke.py` | `f44ff9974784debb836fbac0ad7e601c691b0f60ec416e47ace643fbcd00d87a` |
| `tests/test_continuation_policy_ablation.py` | `c429a8fd957da005d0d0abdbe09ebaa980b4baab8a8407174b83653e805bca60` |
| `configs/continuation_policy_causal_ablation_v1.json` | `881210c980b3d5615914823a5aa08922f02dd8e660f0dc9c7d7b180ef9ee2400` |

Before any Wulver sync, recompute these with `sha256sum` and require an exact
match; a mismatch means the local payload changed since this manifest was
last updated and the manifest must be refreshed first.

## Documentation references -- living docs, not pinned payload

The files below are mutable status/interpretation docs, not sync payload.
They are expected to drift as the branch progresses (this pass alone edited
several of them). Do not treat their hashes as an integrity gate; they are
listed only so a reader knows which docs described the continuation work at
the time this manifest was last touched. Recompute at read time if currency
matters:

| File | SHA256 as of 2026-08-10 (informational only) |
|---|---|
| `docs/reviewer/kbs_negative_results_interpretation.md` | `3fcdc423acd70bb1aa40fcf52f681238bba507c1f4c713e22ba6b6b4396539ac` |
| `docs/reviewer_revision_roadmap.md` | `831041e235a500821264f78583c619dadd467c5564553de68f7b594a6118888d` |
| `docs/reviewer/kbs_second_revision_artifact_map.md` | `62fce8c09cfa4604aaf59e58ef11cab56ea7492fc56ae5da834e7c9f01e0ec32` |
| `docs/reviewer/local_to_wulver_continuation_sync_manifest.md` (this file) | not applicable -- self-hash is not stable when embedded; compute at transfer time with `sha256sum` if needed |

Also relevant but not part of the original payload list: the consolidated
[`KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`](KBS_SECOND_REVISION_HYPOTHESIS_MAP.md)
and
[`KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`](KBS_SECOND_REVISION_REVIEWER_COVERAGE.md)
now hold the continuation-policy hypothesis (H5) and reviewer-concern
(Major 3 / R3) status; sync them as living docs under the same
not-pinned-payload treatment if a Wulver reader needs current status text.

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
